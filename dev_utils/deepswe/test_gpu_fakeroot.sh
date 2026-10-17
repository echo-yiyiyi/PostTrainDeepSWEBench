#!/bin/bash
set -euo pipefail
source src/deepswe/common.sh
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-gpu-check.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 -u dev_utils/deepswe/test_gpu_fakeroot.py
python3 -u dev_utils/deepswe/test_prompt_transport.py
