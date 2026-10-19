#!/bin/bash
set -euo pipefail
source cluster/ibex/deepswe/common.sh
export DEEPSWE_MODE=full
export DEEPSWE_BASE_MODEL=Qwen/Qwen3.5-9B
export DEEPSWE_TRAIN_SECONDS="${DEEPSWE_FULL_TRAIN_SECONDS:-36000}"
export DEEPSWE_RESULTS="$DEEPSWE_STORAGE_RESULTS/deepswe"
export DEEPSWE_TRAIN_SIF="$DEEPSWE_STORAGE_IMAGES/qwen35.sif"
export DEEPSWE_EVAL_SIF="$DEEPSWE_STORAGE_IMAGES/qwen35.sif"
export DEEPSWE_SUITE_MANIFEST="$DEEPSWE_STORAGE_IMAGES/deepswe_manifest.json"
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-full-${SLURM_JOB_ID}.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
if [[ "${1:-train}" == train ]]; then
    python3 cluster/ibex/deepswe/run.py
elif [[ "$1" == evaluate ]]; then
    python3 src/eval/tasks/deepswe/evaluate_final_eval.py "$2" --index "$SLURM_ARRAY_TASK_ID"
elif [[ "$1" == aggregate ]]; then
    python3 src/eval/tasks/deepswe/evaluate_final_eval.py "$2" --aggregate
else
    echo 'Unknown full-suite phase' >&2
    exit 1
fi
