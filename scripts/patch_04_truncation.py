#!/usr/bin/env python
"""One-off maintenance script: make the registered classifier truncate long inputs.

Problem: `mlflow.transformers.log_model` saved the classifier as a plain
text-classification pipeline. At inference the pipeline does NOT truncate, so a
complaint longer than 512 tokens crashes DistilBERT ("size of tensor a (1272)
must match ... (512)"). Training used max_len=256 with truncation, so inference
should do the same. The fix is to log the model with
    model_config={"truncation": True, "max_length": <max_len>}
which MLflow applies to every pipeline call - in-process loading, 06_evaluate,
and `mlflow models serve` alike. That requires re-logging the model, i.e. one
more training run.

Usage (repository root, main environment, MLflow server running):

    python scripts/patch_04_truncation.py                      # 1. add model_config to scripts/04_train_classifier.py
    python scripts/04_train_classifier.py                      # 2. retrain -> new registered version (a few minutes)
    python scripts/patch_04_truncation.py --delete-version 1   # 3. remove the old version that has no truncation config
    python scripts/06_evaluate.py                              # 4. evaluate + set champion
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "04_train_classifier.py"
ANCHOR = 'task="text-classification",'
NEW_LINE = 'model_config={"truncation": True, "max_length": c["max_len"]},  # applied at inference by MLflow'


def patch_training_script() -> None:
    src = TRAIN_SCRIPT.read_text(encoding="utf-8")
    if "model_config=" in src:
        print(f"{TRAIN_SCRIPT.name}: model_config already present, nothing to do")
        return
    lines = src.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if ANCHOR in line:
            indent = re.match(r"\s*", line).group(0)
            lines.insert(i + 1, f"{indent}{NEW_LINE}\n")
            break
    else:
        sys.exit(f"anchor {ANCHOR!r} not found in {TRAIN_SCRIPT}; edit log_model(...) by hand")
    TRAIN_SCRIPT.write_text("".join(lines), encoding="utf-8")
    compile("".join(lines), str(TRAIN_SCRIPT), "exec")  # syntax check
    print(f"{TRAIN_SCRIPT.name}: added model_config to mlflow.transformers.log_model(...)")


def delete_version(version: str) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    import mlflow
    from mlflow import MlflowClient

    from triage.config import CFG

    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
    name = CFG["classifier"]["registered_name"]
    client = MlflowClient()
    for alias in list(client.get_model_version(name, version).aliases):
        client.delete_registered_model_alias(name, alias)
        print(f"removed alias '{alias}' from {name} v{version}")
    client.delete_model_version(name, version)
    print(f"deleted {name} v{version}")
    remaining = sorted(int(v.version) for v in client.search_model_versions(f"name='{name}'"))
    print(f"remaining versions of {name}: {remaining}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete-version", metavar="N", help="delete this registered version of the classifier instead of patching")
    args = ap.parse_args()
    if args.delete_version:
        delete_version(args.delete_version)
    else:
        patch_training_script()


if __name__ == "__main__":
    main()
