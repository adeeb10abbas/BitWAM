#!/usr/bin/env bash
set -euo pipefail

# Supervise the long-running A100 control arms after their first stage starts.
# Usage: run_droid_control_pipeline.sh zero | action

readonly REPO_ROOT="${BITWAM_REPO_ROOT:-/data/users/ali/BitWAM}"
readonly PYTHON_ROOT="${BITVLA_ROOT:-/data/users/ali/BitVLA}"
readonly RUN_ROOT="${BITWAM_RUN_ROOT:-/data/users/ali/bitvla_runs}"
readonly EVAL_ROOT="${BITWAM_EVAL_ROOT:-/data/users/ali/bitvla_evals}"
readonly LOG_ROOT="${BITWAM_LOG_ROOT:-/data/users/ali/logs}"
readonly CONFIG_REVISION="${BITWAM_CONFIG_REVISION:-758dabf}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

process_running() {
  local pid="$1"
  local state
  state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
  [[ -n "${state}" && "${state}" != Z* ]]
}

wait_for_checkpoint() {
  local checkpoint="$1"
  local pid_file="$2"
  local label="$3"
  local pid

  while [[ ! -f "${checkpoint}/bitwam_manifest.json" ]]; do
    if [[ -f "${pid_file}" ]]; then
      pid="$(cat "${pid_file}")"
      if ! process_running "${pid}"; then
        log "ERROR: ${label} exited without its final checkpoint"
        return 1
      fi
    fi
    sleep 30
  done
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}")"
    while process_running "${pid}"; do sleep 30; done
  fi
  log "${label} final checkpoint is ready"
}

validate_checkpoint() {
  local checkpoint="$1"
  local expected_stage="$2"
  local expected_step="$3"
  "${PYTHON_ROOT}/.venv/bin/python" - \
    "${checkpoint}" "${expected_stage}" "${expected_step}" "${CONFIG_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected = {
    "architecture": "native-bitvla-wam",
    "stage": sys.argv[2],
    "step": int(sys.argv[3]),
    "config_revision": sys.argv[4],
}
manifest = json.loads((checkpoint / "bitwam_manifest.json").read_text())
actual = {key: manifest.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"checkpoint manifest mismatch: expected={expected!r} actual={actual!r}")
world_model = checkpoint / f"world_model--{expected['step']}_checkpoint.pt"
if not world_model.is_file() or world_model.stat().st_size == 0:
    raise SystemExit(f"missing world checkpoint: {world_model}")
print(json.dumps({"checkpoint": str(checkpoint), "validated": actual}, sort_keys=True))
PY
}

validate_metrics() {
  local metrics="$1"
  local expected_world_size="$2"
  "${PYTHON_ROOT}/.venv/bin/python" - "${metrics}" "${expected_world_size}" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_world_size = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit(f"empty metrics file: {path}")
required = {
    "schema_version", "world_size", "micro_step", "action_loss", "world_loss",
    "world_cosine_similarity", "world_action_conditioning_gap",
    "global_examples_per_second", "cuda_max_memory_allocated_bytes",
}
for index, row in enumerate(rows):
    if required - row.keys():
        raise SystemExit(f"row {index} missing metrics: {sorted(required - row.keys())}")
    if row["schema_version"] != 2 or row["world_size"] != expected_world_size:
        raise SystemExit(f"row {index} has invalid schema/world size")
    for key, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SystemExit(f"row {index} has non-finite {key}: {value}")
print(json.dumps({
    "metrics": str(path),
    "rows": len(rows),
    "last_micro_step": rows[-1]["micro_step"],
}, sort_keys=True))
PY
}

refuse_existing() {
  local target
  for target in "$@"; do
    if [[ -e "${target}" ]]; then
      log "ERROR: refusing to overwrite existing artifact: ${target}"
      return 1
    fi
  done
}

run_training() {
  local label="$1"
  local config="$2"
  local processes="$3"
  local run_dir="$4"
  local checkpoint="$5"
  local stage="$6"
  local step="$7"
  local log_file="$8"
  local pid_file="$9"

  refuse_existing "${run_dir}" "${checkpoint}"
  log "starting ${label} with ${processes} GPUs"
  (
    cd "${REPO_ROOT}"
    exec env \
      PYTHONPATH="${REPO_ROOT}/src" \
      PYTHONUNBUFFERED=1 \
      TF_CPP_MIN_LOG_LEVEL=2 \
      WANDB_MODE=offline \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PYTHON_ROOT}/.venv/bin/torchrun" \
      --standalone --nproc-per-node="${processes}" \
      --module lerobot_policy_bitwam.bitvla_train \
      --config "${REPO_ROOT}/configs/${config}"
  ) >"${log_file}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" >"${pid_file}"
  wait "${pid}"
  wait_for_checkpoint "${checkpoint}" "${pid_file}" "${label}"
  validate_checkpoint "${checkpoint}" "${stage}" "${step}"
  validate_metrics "${run_dir}/metrics.jsonl" "${processes}"
}

run_holdout() {
  local label="$1"
  local config="$2"
  local run_dir="$3"
  local log_file="$4"

  refuse_existing "${run_dir}"
  log "starting ${label}"
  (
    cd "${REPO_ROOT}"
    exec env \
      PYTHONPATH="${REPO_ROOT}/src" \
      PYTHONUNBUFFERED=1 \
      TF_CPP_MIN_LOG_LEVEL=2 \
      WANDB_MODE=offline \
      "${PYTHON_ROOT}/.venv/bin/torchrun" \
      --standalone --nproc-per-node=1 \
      --module lerobot_policy_bitwam.bitvla_train \
      --config "${REPO_ROOT}/configs/${config}"
  ) >"${log_file}" 2>&1
  validate_metrics "${run_dir}/metrics.jsonl" 1
}

run_eval() {
  local label="$1"
  local config="$2"
  local output_dir="$3"
  local log_file="$4"

  refuse_existing "${output_dir}"
  log "starting ${label}"
  (
    cd "${REPO_ROOT}"
    exec env \
      PYTHONPATH="${REPO_ROOT}/src" \
      PYTHONUNBUFFERED=1 \
      MUJOCO_GL=egl \
      "${PYTHON_ROOT}/.venv/bin/python" \
      -m lerobot_policy_bitwam.bitvla_evaluate \
      --config "${REPO_ROOT}/configs/${config}"
  ) >"${log_file}" 2>&1
  [[ -d "${output_dir}" ]] || {
    log "ERROR: ${label} did not create ${output_dir}"
    return 1
  }
}

zero_pipeline() {
  local zero_checkpoint="${RUN_ROOT}/bitwam-droid-pretrain-zero-action--120000_chkpt"
  local zero_world_size="${BITWAM_ZERO_WORLD_SIZE:-1}"
  if [[ "${BITWAM_EXTERNAL_FIRST_STAGE:-0}" == 1 ]]; then
    while [[ ! -f "${zero_checkpoint}/bitwam_manifest.json" ]]; do
      log "waiting for externally supervised zero-action Stage P"
      sleep 30
    done
  else
    wait_for_checkpoint \
      "${zero_checkpoint}" "${LOG_ROOT}/bitvla-droid-zero-pretrain.pid" \
      "zero-action Stage P"
  fi
  validate_checkpoint \
    "${zero_checkpoint}" droid_frozen_world_pretrain_zero_action 120000
  validate_metrics \
    "${RUN_ROOT}/bitwam-droid-pretrain-zero-action/metrics.jsonl" \
    "${zero_world_size}"

  run_holdout \
    "zero-action-pretrained holdout" \
    bitvla-world-eval-droid-holdout-zero-pretrain.yaml \
    "${RUN_ROOT}/bitwam-droid-holdout-zero-pretrain" \
    "${LOG_ROOT}/bitvla-droid-holdout-zero-pretrain.log"

  local primary_checkpoint="${RUN_ROOT}/bitwam-droid-pretrain--120000_chkpt"
  while [[ ! -f "${primary_checkpoint}/bitwam_manifest.json" ]]; do
    log "waiting for primary Stage P before no-M post-training"
    sleep 30
  done

  run_training \
    "no-M LIBERO post-training" \
    bitvla-world-posttrain-droid-pretrain-libero10.yaml 2 \
    "${RUN_ROOT}/bitwam-droid-pretrain-libero-posttrain" \
    "${RUN_ROOT}/bitwam-droid-pretrain-libero-posttrain--102000_chkpt" \
    droid_pretrain_to_libero_posttrain_no_midtrain 102000 \
    "${LOG_ROOT}/bitvla-droid-no-mid-posttrain.log" \
    "${LOG_ROOT}/bitvla-droid-no-mid-posttrain.pid"
  run_eval \
    "no-M LIBERO-10 evaluation" \
    bitvla-world-posttrain-droid-pretrain-eval-10.yaml \
    "${EVAL_ROOT}/bitwam-droid-pretrain-libero-102000-10" \
    "${LOG_ROOT}/bitvla-droid-no-mid-eval-10.log"
  log "zero/no-M control pipeline completed"
}

action_pipeline() {
  local action_checkpoint="${RUN_ROOT}/bitvla-action-only-droid-midtrain--105000_chkpt"
  wait_for_checkpoint \
    "${action_checkpoint}" "${LOG_ROOT}/bitvla-droid-action-only-midtrain.pid" \
    "action-only Stage M"
  validate_checkpoint "${action_checkpoint}" droid_action_only_midtrain 105000
  validate_metrics "${RUN_ROOT}/bitvla-action-only-droid-midtrain/metrics.jsonl" 2

  run_training \
    "action-only LIBERO post-training" \
    bitvla-action-only-posttrain-droid-libero10.yaml 2 \
    "${RUN_ROOT}/bitvla-action-only-droid-libero-posttrain" \
    "${RUN_ROOT}/bitvla-action-only-droid-libero-posttrain--107000_chkpt" \
    droid_to_libero_action_only_posttrain 107000 \
    "${LOG_ROOT}/bitvla-action-only-posttrain.log" \
    "${LOG_ROOT}/bitvla-action-only-posttrain.pid"
  run_eval \
    "action-only LIBERO-10 evaluation" \
    bitvla-action-only-posttrain-droid-eval-10.yaml \
    "${EVAL_ROOT}/bitvla-action-only-droid-libero-107000-10" \
    "${LOG_ROOT}/bitvla-action-only-eval-10.log"

  local shuffled_checkpoint="${RUN_ROOT}/bitwam-droid-pretrain-shuffled-action--120000_chkpt"
  local shuffled_world_size="${BITWAM_SHUFFLED_WORLD_SIZE:-1}"
  if [[ "${BITWAM_EXTERNAL_SHUFFLED:-0}" == 1 ]]; then
    while [[ ! -f "${shuffled_checkpoint}/bitwam_manifest.json" ]]; do
      log "waiting for externally supervised shuffled-action Stage P"
      sleep 30
    done
    validate_checkpoint \
      "${shuffled_checkpoint}" droid_frozen_world_pretrain_shuffled_action 120000
    validate_metrics \
      "${RUN_ROOT}/bitwam-droid-pretrain-shuffled-action/metrics.jsonl" \
      "${shuffled_world_size}"
  else
    run_training \
      "shuffled-action Stage P" \
      bitvla-world-pretrain-droid-shuffled-action.yaml 1 \
      "${RUN_ROOT}/bitwam-droid-pretrain-shuffled-action" \
      "${shuffled_checkpoint}" \
      droid_frozen_world_pretrain_shuffled_action 120000 \
      "${LOG_ROOT}/bitvla-droid-shuffled-pretrain.log" \
      "${LOG_ROOT}/bitvla-droid-shuffled-pretrain.pid"
  fi
  run_holdout \
    "shuffled-action-pretrained holdout" \
    bitvla-world-eval-droid-holdout-shuffled-pretrain.yaml \
    "${RUN_ROOT}/bitwam-droid-holdout-shuffled-pretrain" \
    "${LOG_ROOT}/bitvla-droid-holdout-shuffled-pretrain.log"
  log "action/shuffled control pipeline completed"
}

case "${1:-}" in
  zero) zero_pipeline ;;
  action) action_pipeline ;;
  *) echo "usage: $0 zero|action" >&2; exit 2 ;;
esac
