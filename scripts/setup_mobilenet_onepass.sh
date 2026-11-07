#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Run from the repository root on the Linux/V100 training server.
Assembles and extracts data/tiny-imagenet-200.zip from the eight uploaded parts,
installs all dependencies into .venv, then checks the real CUDA kernels.
An existing isolated .venv is reused. To create one, the bootstrap interpreter
defaults to python3; set PRUNING_BOOTSTRAP_PYTHON=python3.12 if needed.
This command does not start training.
HELP
    exit 0
fi

[[ -f configs/mobilenet_onepass_thr05_gap05_1pp.yaml ]] || {
    echo "Run this command from the repository root on the new MobileNet branch." >&2
    exit 1
}
if [[ ! -x .venv/bin/python ]]; then
    "${PRUNING_BOOTSTRAP_PYTHON:-python3}" -m venv .venv
fi

.venv/bin/python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required; the prior server used Python 3.12.9.")
if sys.prefix == sys.base_prefix:
    raise SystemExit("Expected an isolated .venv interpreter.")
print("Python:", sys.version.split()[0], sys.executable, flush=True)
PY

.venv/bin/python scripts/prepare_tinyimagenet.py
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements-pruning-v100.txt

# Audio is not used here; keep it compatible if it was already installed.
if .venv/bin/python - <<'PY'
from importlib.util import find_spec
raise SystemExit(0 if find_spec("torchaudio") is not None else 1)
PY
then
    .venv/bin/python -m pip install 'torchaudio==2.6.0+cu126' \
        --index-url https://download.pytorch.org/whl/cu126
fi
.venv/bin/python -m pip check

.venv/bin/python - <<'PY'
import hydra
import mlflow
import numpy
import omegaconf
import torch
import torchvision

assert torch.__version__.split("+")[0] == "2.6.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.21.0", torchvision.__version__
assert torch.cuda.is_available(), "CUDA is unavailable; check the GPU allocation/driver."
capability = torch.cuda.get_device_capability(0)
if capability == (7, 0):
    assert "sm_70" in torch.cuda.get_arch_list(), "This wheel does not support V100 sm_70."
x = torch.randn(256, 256, device="cuda:0")
assert torch.isfinite(x @ x).all().item()
boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda:0")
scores = torch.tensor([1.], device="cuda:0")
assert torchvision.ops.nms(boxes, scores, 0.5).numel() == 1
torch.cuda.synchronize()
print("Torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0), "capability:", capability)
print("CUDA matrix multiplication and torchvision NMS: PASS")
print("Dependencies and dataset are ready. Training has not started.")
PY
