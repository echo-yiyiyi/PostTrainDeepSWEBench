#!/bin/bash
# Prepare CPU-only agent dependencies on the submission node, outside GPU time.
set -euo pipefail
source "$(dirname "$0")/common.sh"
if [[ ! -x "$DEEPSWE_MINI_PYTHON" ]]; then
    uv venv --python 3.12 "$(dirname "$(dirname "$DEEPSWE_MINI_PYTHON")")"
fi
uv pip sync --python "$DEEPSWE_MINI_PYTHON" src/deepswe/mini_swe/requirements.lock
"$DEEPSWE_MINI_PYTHON" -c 'from minisweagent.agents.default import DefaultAgent; import inspect_ai; print("mini-swe evaluator ready")'
