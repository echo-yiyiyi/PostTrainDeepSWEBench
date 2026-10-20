#!/bin/bash
# Re-evaluate an existing checkpoint; no Codex call or training.
set -euo pipefail
source "$(dirname "$0")/../../cluster/ibex/deepswe/common.sh"
[[ $# -eq 1 ]] || { echo "Usage: $0 results/deepswe_smoke/RUN_ID" >&2; exit 1; }
source_run="$(realpath -- "$1")"
[[ -f "$source_run/training_home/task/final_model/config.json" ]] || {
    echo 'Source run has no final_model/config.json' >&2; exit 1;
}
mkdir -p logs
args=(--job-name=deepswe-reeval --nodes=1 --ntasks=1 --cpus-per-task=8
      --mem=64G --gres="${DEEPSWE_GRES:-gpu:a100:1}" --time="${DEEPSWE_EVAL_WALLTIME:-01:30:00}"
      --chdir="$DEEPSWE_ROOT" --output=logs/deepswe_reeval_%j.out
      --error=logs/deepswe_reeval_%j.err)
[[ -z "${DEEPSWE_ACCOUNT:-}" ]] || args+=(--account="$DEEPSWE_ACCOUNT")
[[ -z "${DEEPSWE_PARTITION:-}" ]] || args+=(--partition="$DEEPSWE_PARTITION")
sbatch "${args[@]}" cluster/ibex/deepswe/reevaluate_job.sh "$source_run"
