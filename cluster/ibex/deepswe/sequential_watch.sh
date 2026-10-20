#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
exec python3 -u cluster/ibex/deepswe/sequential_watch.py "$@"
