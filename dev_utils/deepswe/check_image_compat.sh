#!/bin/bash
set -euo pipefail
source cluster/ibex/deepswe/common.sh
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-compat.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 -u dev_utils/deepswe/check_image_compat.py "$1"
