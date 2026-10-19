#!/bin/bash
set -euo pipefail
source src/deepswe/common.sh
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-shell-${SLURM_JOB_ID}.XXXXXX)"
export LITELLM_LOCAL_MODEL_COST_MAP=True
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
"$DEEPSWE_MINI_PYTHON" -u dev_utils/deepswe/test_shell_transport.py --image-smoke
