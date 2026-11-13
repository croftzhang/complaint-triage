#!/usr/bin/env bash
# Run the whole pipeline on synthetic data with tiny models (CPU, a few minutes).
# Prerequisites: MLflow server running (scripts/00_start_mlflow.sh), MAIN env activated,
# and the TF env at ./.venv-tf (or set TF_PYTHON to its python).
#
#   bash scripts/run_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export TRIAGE_CONFIG=config.smoke.yaml
export TRIAGE_FAKE_LLM="${TRIAGE_FAKE_LLM:-1}"
TF_PYTHON="${TF_PYTHON:-.venv-tf/bin/python}"

echo "== 1/6 synthetic data";            python scripts/01_download_cfpb.py --synthetic
echo "== 2/6 Reg E fixture";             python scripts/02_fetch_reg_e.py
echo "== 3/6 embeddings + FAISS";        python scripts/03_embed_and_index.py
echo "== 4/6 PyTorch classifier";        python scripts/04_train_classifier.py
echo "== 5/6 TensorFlow relief model";   "$TF_PYTHON" scripts/05_train_relief_tf.py
echo "== 6/6 evaluate + promote";        python scripts/06_evaluate.py
echo "== demo (fake LLM, auto-approve)"; python scripts/08_demo.py --auto-approve --product-group "Checking or savings account" \
  --text "Someone used my debit card for \$800 in ATM withdrawals I did not authorize. I reported it two days later and the bank refused a refund."
echo "smoke test passed"
