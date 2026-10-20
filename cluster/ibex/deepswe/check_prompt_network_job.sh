#!/bin/bash
set -euo pipefail
source cluster/ibex/deepswe/common.sh
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-network-${SLURM_JOB_ID}.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 dev_utils/deepswe/check_prompt_network.py
