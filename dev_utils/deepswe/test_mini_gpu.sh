#!/bin/bash
set -euo pipefail
source cluster/ibex/deepswe/common.sh
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-mini-${SLURM_JOB_ID}.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 -u dev_utils/deepswe/test_mini_gpu.py "$1"
