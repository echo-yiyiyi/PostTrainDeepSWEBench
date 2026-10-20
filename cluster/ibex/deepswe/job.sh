#!/bin/bash
set -euo pipefail
# sbatch copies this script into its spool; --chdir sets the repository root.
source cluster/ibex/deepswe/common.sh
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Run through commit_deepswe_smoke.sh / sbatch.' >&2; exit 1; }
export DEEPSWE_JOB_WALLTIME="${DEEPSWE_SMOKE_WALLTIME:-${DEEPSWE_JOB_WALLTIME:-01:30:00}}"
export DEEPSWE_MODE=smoke
export DEEPSWE_BASE_MODEL=Qwen/Qwen3.5-9B
export DEEPSWE_TRAIN_SECONDS=3600
export DEEPSWE_RESULTS="$DEEPSWE_STORAGE_RESULTS/deepswe_smoke"
export DEEPSWE_TRAIN_SIF="$DEEPSWE_STORAGE_IMAGES/qwen35_v032.sif"
export DEEPSWE_EVAL_SIF="$DEEPSWE_STORAGE_IMAGES/qwen35_v032.sif"
export DEEPSWE_SUITE_MANIFEST="$DEEPSWE_STORAGE_IMAGES/deepswe_manifest.json"
python3 cluster/ibex/deepswe/check_full_ready.py
# Host Python orchestrates sibling Singularity executions, never nested containers.
# Fakeroot SIF extraction on the shared home filesystem can take minutes.
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-${SLURM_JOB_ID}.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 cluster/ibex/deepswe/run.py
