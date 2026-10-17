#!/bin/bash
set -euo pipefail
source src/deepswe/common.sh
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit this script with sbatch.' >&2; exit 1; }
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-${SLURM_JOB_ID}.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 src/deepswe/reevaluate.py "$1"
