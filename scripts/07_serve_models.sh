#!/usr/bin/env bash
# Step 7 - Serve the registered models with MLflow (one process per model).
#
#   bash scripts/07_serve_models.sh classifier   # run in the MAIN env  (torch/transformers) -> port 5001
#   bash scripts/07_serve_models.sh relief       # run in the TF env    (tensorflow)         -> port 5002
#
# Then set in config.yaml:
#   serving.classifier_backend: http
#   serving.relief_backend: http
#
# Test:
#   curl -s -X POST http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' \
#        -d '{"inputs": ["My debit card was charged for a purchase I did not make."]}'
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

CONFIG_FILE="${TRIAGE_CONFIG:-config.yaml}"
case "${1:-}" in
  classifier)
    NAME=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG_FILE'))['classifier']['registered_name'])")
    PORT=5001 ;;
  relief)
    NAME=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG_FILE'))['relief']['registered_name'])")
    PORT=5002 ;;
  *) echo "usage: $0 classifier|relief"; exit 1 ;;
esac

echo "serving models:/${NAME}@champion on port ${PORT} (env-manager=local: uses the current virtualenv)"
exec mlflow models serve -m "models:/${NAME}@champion" -p "${PORT}" --host 127.0.0.1 --env-manager local
