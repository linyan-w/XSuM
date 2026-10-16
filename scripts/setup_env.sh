#!/usr/bin/env bash
# Create/refresh the project virtualenv from Python 3.9 and install pinned dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 9), sys.version' || { echo "need python >= 3.9"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
.venv/bin/python -c 'import tables, pandas; print("pandas", pandas.__version__, "pytables", tables.__version__)'
echo "activate with: source .venv/bin/activate"
