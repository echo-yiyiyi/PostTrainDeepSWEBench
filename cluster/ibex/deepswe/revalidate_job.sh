#!/bin/bash
# Validate the active image against the current prompt/evaluation protocol.
set -euo pipefail
source cluster/ibex/deepswe/common.sh
export SINGULARITY_TMPDIR="$(mktemp -d /tmp/deepswe-revalidate-${SLURM_JOB_ID}.XXXXXX)"
export DEEPSWE_CONTAINER_TMPDIR="$SINGULARITY_TMPDIR"
trap 'rm -rf -- "$SINGULARITY_TMPDIR"' EXIT
image="$DEEPSWE_STORAGE_IMAGES/qwen35_v032.sif"
[[ -f "$image" ]] || { echo "Validated image is missing: $image" >&2; exit 1; }
python3 dev_utils/deepswe/check_qwen35.py "$image"
ready="$DEEPSWE_STORAGE_IMAGES/qwen35_v032_ready.json"
if [[ -f "$ready" ]]; then
    cp "$ready" "$DEEPSWE_STORAGE_RESULTS/qwen35_check_${SLURM_JOB_ID}/previous_ready.json"
fi
cp "$DEEPSWE_STORAGE_RESULTS/qwen35_check_${SLURM_JOB_ID}/ready.json" "${ready}.tmp.${SLURM_JOB_ID}"
mv "${ready}.tmp.${SLURM_JOB_ID}" "$ready"
python3 cluster/ibex/deepswe/check_full_ready.py
