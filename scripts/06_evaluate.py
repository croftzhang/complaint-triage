#!/usr/bin/env python
"""Step 6 - Evaluate and promote models (main environment).

  1. Classifier: re-score the newest registered version on the test set through
     the MLflow pyfunc wrapper (this also proves the logged artifact works),
     then set alias 'champion' on the version with the best test_macro_f1.
  2. Retrieval: recall@k on evals/retrieval_questions.jsonl.
  3. Relief model: read the metrics logged by step 5 (the TF model is not loaded
     here - it lives in the TF environment) and set its 'champion' alias.
  4. Write reports/eval_report.md and log everything to an 'evaluation' MLflow run.

  python scripts/06_evaluate.py
  python scripts/06_evaluate.py --max-rows 500     # faster classifier re-scoring
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.config import CFG, REPO_ROOT, path  # noqa: E402
from triage.data_io import load_split  # noqa: E402
from triage.tracking import set_champion, setup_mlflow, versions_with_metrics  # noqa: E402


def eval_classifier(max_rows: int | None) -> dict:
    name = CFG["classifier"]["registered_name"]
    rows = versions_with_metrics(name, ["test_accuracy", "test_macro_f1"])
    if not rows:
        return {"skipped": "no registered classifier"}
    best = max(rows, key=lambda r: r["test_macro_f1"] or -1)
    set_champion(name, best["version"])

    test = load_split(path("processed_dir"), "test")
    if max_rows:
        test = test.sample(n=min(max_rows, len(test)), random_state=CFG["data"]["seed"])
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    pred = model.predict(test["narrative"].tolist())
    labels = pred["label"].tolist() if hasattr(pred, "iloc") else [p["label"] for p in pred]
    acc = accuracy_score(test["product_group"], labels)
    f1 = f1_score(test["product_group"], labels, average="macro")
    per_class = (
        pd.DataFrame({"y": test["product_group"].to_numpy(), "p": labels})
        .assign(hit=lambda d: d.y == d.p).groupby("y")["hit"].mean().round(3).to_dict()
    )
    return {"versions": rows, "champion_version": best["version"], "n_eval": len(test),
            "accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4), "per_class_accuracy": per_class}


def eval_retrieval() -> dict:
    from triage.retrieval import retrieve

    k = CFG["retrieval"]["k"]
    qs = [json.loads(l) for l in (REPO_ROOT / "evals/retrieval_questions.jsonl").read_text().splitlines() if l.strip()]
    hits, misses = 0, []
    for q in qs:
        got = [d.metadata["section"] for d in retrieve(q["question"], k)]
        ok = any(s in got for s in q["expected_sections"])
        hits += ok
        if not ok:
            misses.append({"question": q["question"], "expected": q["expected_sections"], "got": got})
    return {"n": len(qs), "k": k, f"recall@{k}": round(hits / len(qs), 4), "misses": misses}


def eval_relief() -> dict:
    name = CFG["relief"]["registered_name"]
    rows = versions_with_metrics(name, ["test_pr_auc", "test_roc_auc", "test_precision_at_thr", "test_recall_at_thr"])
    if not rows:
        return {"skipped": "no registered relief model (run scripts/05 in the TF env)"}
    best = max(rows, key=lambda r: r["test_pr_auc"] or -1)
    set_champion(name, best["version"])
    return {"versions": rows, "champion_version": best["version"], **{k: v for k, v in best.items() if k.startswith("test_")}}


def write_report(results: dict, out: Path) -> None:
    c, r, l = results["classifier"], results["retrieval"], results["relief"]
    lines = ["# Evaluation report", ""]
    lines += ["## Classifier (PyTorch + HuggingFace)", ""]
    if "skipped" in c:
        lines.append(c["skipped"])
    else:
        lines += [f"- champion version: {c['champion_version']}  (evaluated on {c['n_eval']} test rows)",
                  f"- accuracy: **{c['accuracy']}**", f"- macro F1: **{c['macro_f1']}**", "", "| product group | accuracy |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in c["per_class_accuracy"].items()]
    recall_key = "recall@%d" % r["k"]
    lines += ["", "## Retrieval (FAISS over Regulation E)", "",
              f"- {recall_key}: **{r[recall_key]}** on {r['n']} questions"]
    if r["misses"]:
        lines += ["", "Misses:", ""] + [f"- {m['question']}  expected {m['expected']}, got {m['got']}" for m in r["misses"]]
    lines += ["", "## Relief model (TensorFlow)", ""]
    if "skipped" in l:
        lines.append(l["skipped"])
    else:
        lines += [f"- champion version: {l['champion_version']}", f"- test PR-AUC: **{l['test_pr_auc']:.4f}**",
                  f"- test ROC-AUC: {l['test_roc_auc']:.4f}", f"- precision / recall @ threshold: {l['test_precision_at_thr']:.3f} / {l['test_recall_at_thr']:.3f}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    setup_mlflow()
    results = {"classifier": eval_classifier(args.max_rows), "retrieval": eval_retrieval(), "relief": eval_relief()}

    with mlflow.start_run(run_name="evaluation"):
        mlflow.set_tag("model_type", "evaluation")
        if "accuracy" in results["classifier"]:
            mlflow.log_metrics({"clf_accuracy": results["classifier"]["accuracy"], "clf_macro_f1": results["classifier"]["macro_f1"]})
        k = results["retrieval"]["k"]
        mlflow.log_metric(f"retrieval_recall_at_{k}", results["retrieval"][f"recall@{k}"])
        if "test_pr_auc" in results["relief"]:
            mlflow.log_metric("relief_test_pr_auc", results["relief"]["test_pr_auc"])
        report = path("reports_dir") / "eval_report.md"
        write_report(results, report)
        (path("reports_dir") / "eval_results.json").write_text(json.dumps(results, indent=2, default=str))
        mlflow.log_artifact(str(report))


if __name__ == "__main__":
    main()
