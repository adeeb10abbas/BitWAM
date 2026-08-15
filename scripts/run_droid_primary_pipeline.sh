#!/usr/bin/env bash
set -euo pipefail

# Continue the preregistered primary DROID study after Stage P has started.
# The launcher is deliberately fail-closed: it validates every artifact and
# refuses to overwrite an existing downstream run.

readonly REPO_ROOT="${BITWAM_REPO_ROOT:-/data/users/ali/BitWAM}"
readonly PYTHON_ROOT="${BITVLA_ROOT:-/data/users/ali/BitVLA}"
readonly RUN_ROOT="${BITWAM_RUN_ROOT:-/data/users/ali/bitvla_runs}"
readonly EVAL_ROOT="${BITWAM_EVAL_ROOT:-/data/users/ali/bitvla_evals}"
readonly LOG_ROOT="${BITWAM_LOG_ROOT:-/data/users/ali/logs}"
readonly CONFIG_REVISION="${BITWAM_CONFIG_REVISION:-821b1bf}"
readonly PRIMARY_PID_FILE="${BITWAM_PRIMARY_PID_FILE:-${LOG_ROOT}/bitvla-droid-pretrain-after-smoke.pid}"
readonly PRIMARY_CHECKPOINT="${RUN_ROOT}/bitwam-droid-pretrain--120000_chkpt"
readonly PIPELINE_SUMMARY="${RUN_ROOT}/bitwam-droid-study-summary.json"

mkdir -p "${LOG_ROOT}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

process_running() {
  local pid="$1"
  local state
  state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
  [[ -n "${state}" && "${state}" != Z* ]]
}

wait_for_process_exit() {
  local pid_file="$1"
  local label="$2"
  local pid
  pid="$(cat "${pid_file}")"
  while process_running "${pid}"; do
    sleep 30
  done
  log "${label} process ${pid} exited"
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
        log "ERROR: ${label} exited without ${checkpoint}/bitwam_manifest.json"
        return 1
      fi
    fi
    sleep 30
  done
  log "${label} checkpoint is present: ${checkpoint}"
}

validate_checkpoint() {
  local checkpoint="$1"
  local expected_stage="$2"
  local expected_step="$3"
  "${PYTHON_ROOT}/.venv/bin/python" - "${checkpoint}" "${expected_stage}" \
    "${expected_step}" "${CONFIG_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
expected_stage = sys.argv[2]
expected_step = int(sys.argv[3])
expected_revision = sys.argv[4]
manifest = json.loads((checkpoint / "bitwam_manifest.json").read_text())
expected = {
    "architecture": "native-bitvla-wam",
    "stage": expected_stage,
    "config_revision": expected_revision,
    "step": expected_step,
}
actual = {key: manifest.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"checkpoint manifest mismatch: expected={expected!r} actual={actual!r}")
world_model = checkpoint / f"world_model--{expected_step}_checkpoint.pt"
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
    "schema_version",
    "world_size",
    "micro_step",
    "action_loss",
    "world_loss",
    "world_cosine_similarity",
    "world_action_conditioning_gap",
    "global_examples_per_second",
    "cuda_max_memory_allocated_bytes",
}
for index, row in enumerate(rows):
    missing = required - row.keys()
    if missing:
        raise SystemExit(f"row {index} missing metrics: {sorted(missing)}")
    if row["schema_version"] != 2 or row["world_size"] != expected_world_size:
        raise SystemExit(f"row {index} has invalid schema/world size: {row}")
    for key, value in row.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SystemExit(f"row {index} has non-finite {key}: {value}")
print(json.dumps({
    "metrics": str(path),
    "rows": len(rows),
    "first_micro_step": rows[0]["micro_step"],
    "last_micro_step": rows[-1]["micro_step"],
    "last": {key: rows[-1][key] for key in sorted(required)},
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

run_training_stage() {
  local label="$1"
  local config="$2"
  local processes="$3"
  local run_dir="$4"
  local checkpoint="$5"
  local expected_stage="$6"
  local expected_step="$7"
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
      "${PYTHON_ROOT}/.venv/bin/torchrun" \
      --standalone \
      --nproc-per-node="${processes}" \
      --module lerobot_policy_bitwam.bitvla_train \
      --config "${REPO_ROOT}/configs/${config}"
  ) >"${log_file}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" >"${pid_file}"
  wait "${pid}"
  wait_for_checkpoint "${checkpoint}" "${pid_file}" "${label}"
  validate_checkpoint "${checkpoint}" "${expected_stage}" "${expected_step}"
  validate_metrics "${run_dir}/metrics.jsonl" "${processes}"
  log "completed ${label}"
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
      --standalone \
      --nproc-per-node=1 \
      --module lerobot_policy_bitwam.bitvla_train \
      --config "${REPO_ROOT}/configs/${config}"
  ) >"${log_file}" 2>&1
  validate_metrics "${run_dir}/metrics.jsonl" 1
  log "completed ${label}"
}

write_summary() {
  "${PYTHON_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/summarize_droid_study.py" \
    --run-root "${RUN_ROOT}" \
    --output "${PIPELINE_SUMMARY}"
}

require_gate() {
  local gate="$1"
  write_summary
  "${PYTHON_ROOT}/.venv/bin/python" - "${PIPELINE_SUMMARY}" "${gate}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
gate_name = sys.argv[2]
gate = summary["gates"][gate_name]
if gate.get("status") != "passed":
    raise SystemExit(f"promotion gate {gate_name} did not pass: {gate}")
print(json.dumps({"gate": gate_name, **gate}, sort_keys=True))
PY
}

run_closed_loop_eval() {
  local run_dir="${EVAL_ROOT}/bitwam-droid-libero-107000-10"
  refuse_existing "${run_dir}"
  log "starting 10-episode-per-task LIBERO-10 evaluation"
  (
    cd "${REPO_ROOT}"
    exec env \
      PYTHONPATH="${REPO_ROOT}/src" \
      PYTHONUNBUFFERED=1 \
      MUJOCO_GL=egl \
      "${PYTHON_ROOT}/.venv/bin/python" \
      -m lerobot_policy_bitwam.bitvla_evaluate \
      --config "${REPO_ROOT}/configs/bitvla-world-posttrain-droid-eval-10.yaml"
  ) >"${LOG_ROOT}/bitvla-droid-posttrain-eval-10.log" 2>&1
  [[ -d "${run_dir}" ]] || {
    log "ERROR: evaluation did not create ${run_dir}"
    return 1
  }
  log "completed primary closed-loop evaluation: ${run_dir}"
}

main() {
  log "waiting for primary Stage P"
  wait_for_checkpoint "${PRIMARY_CHECKPOINT}" "${PRIMARY_PID_FILE}" "Stage P"
  wait_for_process_exit "${PRIMARY_PID_FILE}" "Stage P"
  validate_checkpoint \
    "${PRIMARY_CHECKPOINT}" droid_frozen_world_pretrain 120000
  validate_metrics "${RUN_ROOT}/bitwam-droid-pretrain/metrics.jsonl" 4

  run_holdout \
    "Stage P normal-action holdout" \
    bitvla-world-eval-droid-holdout-pretrain.yaml \
    "${RUN_ROOT}/bitwam-droid-holdout-pretrain-normal" \
    "${LOG_ROOT}/bitvla-droid-holdout-pretrain-normal.log"
  run_holdout \
    "Stage P zero-action holdout" \
    bitvla-world-eval-droid-holdout-pretrain-zero-action.yaml \
    "${RUN_ROOT}/bitwam-droid-holdout-pretrain-zero" \
    "${LOG_ROOT}/bitvla-droid-holdout-pretrain-zero.log"
  run_holdout \
    "Stage P shuffled-action holdout" \
    bitvla-world-eval-droid-holdout-pretrain-shuffled-action.yaml \
    "${RUN_ROOT}/bitwam-droid-holdout-pretrain-shuffled" \
    "${LOG_ROOT}/bitvla-droid-holdout-pretrain-shuffled.log"

  while [[ ! -s "${RUN_ROOT}/bitwam-droid-holdout-initialization/metrics.jsonl" ]]; do
    log "waiting for initialization holdout metrics"
    sleep 30
  done
  validate_metrics "${RUN_ROOT}/bitwam-droid-holdout-initialization/metrics.jsonl" 1
  require_gate stage_p

  run_training_stage \
    "Stage M joint DROID mid-training" \
    bitvla-world-midtrain-droid.yaml 4 \
    "${RUN_ROOT}/bitwam-droid-midtrain" \
    "${RUN_ROOT}/bitwam-droid-midtrain--105000_chkpt" \
    droid_joint_action_world_midtrain 105000 \
    "${LOG_ROOT}/bitvla-droid-midtrain.log" \
    "${LOG_ROOT}/bitvla-droid-midtrain.pid"
  require_gate stage_m

  run_training_stage \
    "Stage F ternary LIBERO post-training" \
    bitvla-world-posttrain-droid-libero10.yaml 4 \
    "${RUN_ROOT}/bitwam-droid-libero-posttrain" \
    "${RUN_ROOT}/bitwam-droid-libero-posttrain--107000_chkpt" \
    droid_to_libero_joint_posttrain 107000 \
    "${LOG_ROOT}/bitvla-droid-posttrain.log" \
    "${LOG_ROOT}/bitvla-droid-posttrain.pid"

  run_closed_loop_eval
  write_summary
  log "primary DROID pipeline completed"
}

main "$@"
