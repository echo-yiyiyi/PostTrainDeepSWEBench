#!/bin/bash
set -euo pipefail
# sbatch copies this script into its spool; --chdir sets the repository root.
source cluster/ibex/deepswe/common.sh
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run through commit_deepswe_smoke.sh / sbatch.' >&2; exit 1; }
# Host Python orchestrates sibling Singularity executions, never nested containers.
# Fakeroot SIF extraction on the shared home filesystem can take minutes.
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-${SLURM_JOB_ID}.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 cluster/ibex/deepswe/run.py
