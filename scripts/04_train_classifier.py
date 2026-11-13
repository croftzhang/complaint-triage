#!/usr/bin/env python
"""Step 4 - Fine-tune a HuggingFace encoder with PyTorch to classify complaints
into 9 product groups, tracking everything in MLflow and registering the model.

  python scripts/04_train_classifier.py
  python scripts/04_train_classifier.py --epochs 1 --max-rows 2000     # quick run

MLflow records: hyper-parameters, per-step training loss, per-epoch eval metrics
(via the HF Trainer MLflow callback), final test metrics, the confusion matrix,
and the model itself (transformers flavor) as a new registered version.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, path, resolve  # noqa: E402
from triage.data_io import load_split  # noqa: E402
from triage.taxonomy import ID2LABEL, LABEL2ID, LABELS  # noqa: E402
from triage.tracking import setup_mlflow  # noqa: E402


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "macro_f1": f1_score(labels, preds, average="macro")}


def main() -> None:
    c = CFG["classifier"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=c["epochs"])
    ap.add_argument("--max-rows", type=int, default=None, help="subsample training rows (quick runs)")
    args = ap.parse_args()

    processed_dir = path("processed_dir")
    train_df, test_df = load_split(processed_dir, "train"), load_split(processed_dir, "test")
    if args.max_rows:
        train_df = train_df.sample(n=min(args.max_rows, len(train_df)), random_state=CFG["data"]["seed"])
    if CFG.get("smoke", {}).get("tiny_models"):
        from triage.smoke import ensure_tiny_models
        ensure_tiny_models(CFG, train_df)

    base_model = resolve(c["base_model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(c.get("bf16")) and device == "cuda"
    print(f"base model: {base_model} | device: {device} | bf16: {use_bf16} | train rows: {len(train_df)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    def to_dataset(df):
        ds = Dataset.from_dict({"text": df["narrative"].tolist(), "label": df["product_group"].map(LABEL2ID).tolist()})
        return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=c["max_len"]), batched=True, remove_columns=["text"])

    train_ds, test_ds = to_dataset(train_df), to_dataset(test_df)

    setup_mlflow()
    out_dir = path("processed_dir").parent / "checkpoints" / "classifier"
    with mlflow.start_run(run_name=f"classifier-{Path(base_model).name}") as run:
        mlflow.set_tags({"model_type": "classifier", "framework": "pytorch+transformers"})
        mlflow.log_params({"base_model": base_model, "max_len": c["max_len"], "n_train": len(train_df), "n_test": len(test_df), "device": device})

        total_steps = int(np.ceil(len(train_ds) / c["batch_size"]) * args.epochs)
        targs = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=c["batch_size"],
            per_device_eval_batch_size=c["batch_size"] * 2,
            num_train_epochs=args.epochs,
            learning_rate=float(c["lr"]),
            weight_decay=0.01,
            warmup_steps=int(0.06 * total_steps),
            bf16=use_bf16,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=25,
            report_to=["mlflow"],          # HF Trainer -> MLflow callback logs loss/eval each step/epoch
            seed=CFG["data"]["seed"],
            dataloader_pin_memory=device == "cuda",
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
        t0 = time.time()
        trainer.train()
        train_seconds = time.time() - t0

        # ---- final test metrics
        pred = trainer.predict(test_ds)
        y_pred, y_true = np.argmax(pred.predictions, axis=-1), np.asarray(pred.label_ids)
        test_metrics = {
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(test_metrics)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
        cm_path = out_dir / "confusion_matrix.json"
        cm_path.write_text(json.dumps({"labels": LABELS, "matrix": cm.tolist()}, indent=2))
        mlflow.log_artifact(str(cm_path))
        print(json.dumps(test_metrics, indent=2))

        # ---- log + register the model (transformers flavor => served by `mlflow models serve`)
        model.to("cpu")
        import accelerate, transformers  # noqa: E401  (pin the serving requirements explicitly)
        info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            name="model",
            task="text-classification",
            model_config={"truncation": True, "max_length": c["max_len"]},  # applied at inference by MLflow
            registered_model_name=c["registered_name"],
            input_example=["My debit card was charged for a purchase I did not make."],
            pip_requirements=[
                f"torch=={torch.__version__.split('+')[0]}",
                f"transformers=={transformers.__version__}",
                f"accelerate=={accelerate.__version__}",
            ],
        )
        (processed_dir / "label_map.json").write_text(json.dumps({"labels": LABELS}, indent=2))
        print(f"registered '{c['registered_name']}' version {info.registered_model_version} (run {run.info.run_id})")


if __name__ == "__main__":
    main()
