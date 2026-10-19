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
# External storage is explicit; defaults remain portable for other checkouts.
export DEEPSWE_STORAGE_RESULTS="$(realpath -m -- "${DEEPSWE_STORAGE_RESULTS:-$DEEPSWE_ROOT/results}")"
export DEEPSWE_STORAGE_IMAGES="$(realpath -m -- "${DEEPSWE_STORAGE_IMAGES:-$DEEPSWE_ROOT/containers}")"
export DEEPSWE_TRAIN_SIF="${DEEPSWE_TRAIN_SIF:-$DEEPSWE_STORAGE_IMAGES/standard.sif}"
export DEEPSWE_EVAL_SIF="${DEEPSWE_EVAL_SIF:-$DEEPSWE_STORAGE_IMAGES/vllm_debug.sif}"
export DEEPSWE_TASK_SIF="${DEEPSWE_TASK_SIF:-$DEEPSWE_STORAGE_IMAGES/deepswe-happy-dom.sif}"
export HF_HOME="${HF_HOME:-$DEEPSWE_STORAGE_RESULTS/.cache/huggingface}"
export DEEPSWE_RESULTS="${DEEPSWE_RESULTS:-$DEEPSWE_STORAGE_RESULTS/deepswe_smoke}"
export DEEPSWE_MINI_PYTHON="${DEEPSWE_MINI_PYTHON:-$DEEPSWE_STORAGE_RESULTS/mini-swe-venv/bin/python}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$DEEPSWE_STORAGE_RESULTS/.cache/singularity-build}"
# Permit this checkout and explicitly configured storage, never an implicit
# fallback to another checkout's configuration or images.
for _deepswe_path_var in DEEPSWE_TRAIN_SIF DEEPSWE_EVAL_SIF DEEPSWE_TASK_SIF HF_HOME DEEPSWE_RESULTS SINGULARITY_CACHEDIR; do
    _deepswe_resolved="$(realpath -m -- "${!_deepswe_path_var}")"
    case "$_deepswe_resolved" in
        "$DEEPSWE_ROOT"/*|"$DEEPSWE_STORAGE_RESULTS"/*|"$DEEPSWE_STORAGE_IMAGES"/*) export "$_deepswe_path_var=$_deepswe_resolved" ;;
        *) echo "$_deepswe_path_var must be inside $DEEPSWE_ROOT or configured DeepSWE storage" >&2; return 1 ;;
    esac
done
unset _deepswe_path_var _deepswe_resolved
