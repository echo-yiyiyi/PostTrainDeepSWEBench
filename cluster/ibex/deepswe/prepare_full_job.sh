#!/bin/bash
set -euo pipefail
source cluster/ibex/deepswe/common.sh
export SINGULARITY_TMPDIR="$(mktemp -d /tmp/deepswe-qwen35-${SLURM_JOB_ID}.XXXXXX)"
export DEEPSWE_CONTAINER_TMPDIR="$SINGULARITY_TMPDIR"
trap 'rm -rf -- "$SINGULARITY_TMPDIR"' EXIT
"$DEEPSWE_MINI_PYTHON" cluster/ibex/deepswe/prepare_full.py
# A failed GPU gate leaves its built candidate for validation-only retries.
# Do not repeatedly rebuild/download a successfully constructed image.
if [[ $# -gt 0 ]]; then
    [[ $# -eq 1 && -f "$1" ]] || { echo 'Usage: prepare_full_job.sh [existing-candidate.sif]' >&2; exit 1; }
    candidate="$(realpath -- "$1")"
    case "$candidate" in
        "$DEEPSWE_STORAGE_IMAGES"/qwen35_v032_build_*.sif) ;;
        *) echo 'Expected an inactive Qwen3.5 candidate in configured image storage' >&2; exit 1 ;;
    esac
else
    candidate="$DEEPSWE_STORAGE_IMAGES/qwen35_v032_build_${SLURM_JOB_ID}.sif"
    singularity build --fakeroot "$candidate" containers/deepswe_qwen35.def
fi
python3 dev_utils/deepswe/check_qwen35.py "$candidate"
# Activate only after real loading, training/export and tool-call checks succeed.
[[ ! -e "$DEEPSWE_STORAGE_IMAGES/qwen35_v032.sif" ]] || { echo 'qwen35_v032.sif already exists; preserve it' >&2; exit 1; }
mv "$candidate" "$DEEPSWE_STORAGE_IMAGES/qwen35_v032.sif"
cp "$DEEPSWE_STORAGE_RESULTS/qwen35_check_${SLURM_JOB_ID}/ready.json" "$DEEPSWE_STORAGE_IMAGES/qwen35_v032_ready.json"
