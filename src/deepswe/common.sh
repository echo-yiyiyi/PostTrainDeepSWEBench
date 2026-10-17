#!/bin/bash
# Source from submit/build/job entry points. Never print credentials.
DEEPSWE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DEEPSWE_ROOT"
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module load "${DEEPSWE_SINGULARITY_MODULE:-singularity/3.9.7}"
# This checkout owns its configuration. Never fall back to another checkout.
DEEPSWE_ENV_FILE="$DEEPSWE_ROOT/.env"
if [[ -f "$DEEPSWE_ENV_FILE" ]]; then
    set -a
    source "$DEEPSWE_ENV_FILE"
    set +a
fi
export DEEPSWE_TRAIN_SIF="${DEEPSWE_TRAIN_SIF:-$DEEPSWE_ROOT/containers/standard.sif}"
export DEEPSWE_EVAL_SIF="${DEEPSWE_EVAL_SIF:-$DEEPSWE_ROOT/containers/vllm_debug.sif}"
export DEEPSWE_TASK_SIF="${DEEPSWE_TASK_SIF:-$DEEPSWE_ROOT/containers/deepswe-fastapi.sif}"
export HF_HOME="${HF_HOME:-$DEEPSWE_ROOT/.cache/huggingface}"
export DEEPSWE_RESULTS="${DEEPSWE_RESULTS:-$DEEPSWE_ROOT/results/deepswe_smoke}"
# Reject stale exported paths and symlinks into another checkout, including
# environments inherited by sbatch from an older submission script.
for _deepswe_path_var in DEEPSWE_TRAIN_SIF DEEPSWE_EVAL_SIF DEEPSWE_TASK_SIF HF_HOME DEEPSWE_RESULTS; do
    _deepswe_resolved="$(realpath -m -- "${!_deepswe_path_var}")"
    case "$_deepswe_resolved" in
        "$DEEPSWE_ROOT"/*) export "$_deepswe_path_var=$_deepswe_resolved" ;;
        *) echo "$_deepswe_path_var must be inside $DEEPSWE_ROOT" >&2; return 1 ;;
    esac
done
unset _deepswe_path_var _deepswe_resolved
