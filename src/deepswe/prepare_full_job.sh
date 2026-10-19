#!/bin/bash
set -euo pipefail
source src/deepswe/common.sh
export SINGULARITY_TMPDIR="$(mktemp -d /tmp/deepswe-qwen35-${SLURM_JOB_ID}.XXXXXX)"
export DEEPSWE_CONTAINER_TMPDIR="$SINGULARITY_TMPDIR"
trap 'rm -rf -- "$SINGULARITY_TMPDIR"' EXIT
"$DEEPSWE_MINI_PYTHON" src/deepswe/prepare_full.py
candidate="$DEEPSWE_STORAGE_IMAGES/qwen35_build_${SLURM_JOB_ID}.sif"
singularity build --fakeroot "$candidate" containers/deepswe_qwen35.def
python3 dev_utils/deepswe/check_qwen35.py "$candidate"
# Activate only after real loading, training/export and tool-call checks succeed.
[[ ! -e "$DEEPSWE_STORAGE_IMAGES/qwen35.sif" ]] || { echo 'qwen35.sif already exists; preserve it' >&2; exit 1; }
mv "$candidate" "$DEEPSWE_STORAGE_IMAGES/qwen35.sif"
cp "$DEEPSWE_STORAGE_RESULTS/qwen35_check_${SLURM_JOB_ID}/ready.json" "$DEEPSWE_STORAGE_IMAGES/qwen35_ready.json"
