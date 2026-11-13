#!/usr/bin/env bash
# Step 0 - Start the MLflow tracking server + model registry (keep this terminal open).
# SQLite backend is required for the Model Registry (the plain file store does not support it).
# UI: http://127.0.0.1:5000
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p mlruns
exec mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlruns \
  --host 127.0.0.1 --port 5000
