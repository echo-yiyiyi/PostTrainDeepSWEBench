#!/bin/bash
set -euo pipefail
# sbatch copies this script into its spool; --chdir sets the repository root.
source src/deepswe/common.sh
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run through commit_deepswe.sh / sbatch.' >&2; exit 1; }
# Host Python orchestrates sibling Singularity executions, never nested containers.
exec python3 src/deepswe/run.py
