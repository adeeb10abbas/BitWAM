#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

usage() {
  cat <<EOF
Usage: ./run_lerobot_examples.sh <command>

Commands:
  setup                     Install package + LeRobot (pip-first)
  gpu_info                  Show nvidia-smi and torch CUDA status
  smoke_pusht               Quick PushT pipeline run
  smoke_aloha               Quick ALOHA sim pipeline run
  smoke_pusht_gpu           Quick PushT run forced on CUDA
  smoke_aloha_gpu           Quick ALOHA run forced on CUDA
  train_pusht               Standard PushT training
  train_aloha               Standard ALOHA sim training
  rollout_pusht             Rollout-style eval for latest PushT quick checkpoint
  rollout_aloha             Rollout-style eval for latest ALOHA quick checkpoint
  test                      Run core tests
EOF
}

check_python() {
  python -c "import torch, bit_vla" >/dev/null
  python -c "import lerobot" >/dev/null
}

check_cuda() {
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in torch. Check drivers, CUDA toolkit, and torch build.")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")
PY
}

gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "nvidia-smi not found in PATH"
  fi
  check_python
  check_cuda
}

setup() {
  pip install -e .
  pip install lerobot
  check_python
  echo "setup complete"
}

smoke_pusht() {
  check_python
  python scripts/train_with_lerobot.py --config configs/pusht_quick.yaml
}

smoke_aloha() {
  check_python
  python scripts/train_with_lerobot.py --config configs/aloha_sim_quick.yaml
}

smoke_pusht_gpu() {
  check_python
  check_cuda
  python scripts/train_with_lerobot.py --config configs/pusht_quick.yaml --device cuda
}

smoke_aloha_gpu() {
  check_python
  check_cuda
  python scripts/train_with_lerobot.py --config configs/aloha_sim_quick.yaml --device cuda
}

train_pusht() {
  check_python
  python scripts/train_with_lerobot.py --config configs/pusht_standard.yaml
}

train_aloha() {
  check_python
  python scripts/train_with_lerobot.py --config configs/aloha_sim_standard.yaml
}

rollout_pusht() {
  check_python
  python scripts/rollout_eval.py \
    --config configs/pusht_quick.yaml \
    --checkpoint outputs/vla_pipeline/lerobot_pusht_quick/best_model.pt
}

rollout_aloha() {
  check_python
  python scripts/rollout_eval.py \
    --config configs/aloha_sim_quick.yaml \
    --checkpoint outputs/vla_pipeline/lerobot_aloha_sim_quick/best_model.pt
}

run_tests() {
  pytest tests/test_models.py tests/test_training.py tests/test_utils.py
}

cmd="${1:-help}"
case "$cmd" in
  setup) setup ;;
  gpu_info) gpu_info ;;
  smoke_pusht) smoke_pusht ;;
  smoke_aloha) smoke_aloha ;;
  smoke_pusht_gpu) smoke_pusht_gpu ;;
  smoke_aloha_gpu) smoke_aloha_gpu ;;
  train_pusht) train_pusht ;;
  train_aloha) train_aloha ;;
  rollout_pusht) rollout_pusht ;;
  rollout_aloha) rollout_aloha ;;
  test) run_tests ;;
  *) usage ;;
esac
