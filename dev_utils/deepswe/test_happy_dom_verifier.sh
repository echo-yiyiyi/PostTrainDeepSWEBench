#!/bin/bash
set -euo pipefail
source cluster/ibex/deepswe/common.sh
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-happy-dom.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 -u dev_utils/deepswe/test_happy_dom_verifier.py
