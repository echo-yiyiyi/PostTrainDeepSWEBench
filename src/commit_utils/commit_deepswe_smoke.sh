#!/bin/bash
# One hour of the full Qwen3.5 training protocol, evaluated on one held-out task.
set -euo pipefail
source "$(dirname "$0")/../../cluster/ibex/deepswe/common.sh"
mkdir -p logs
smoke_walltime="${DEEPSWE_JOB_WALLTIME:-01:30:00}"
dry_run=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --walltime) [[ $# -ge 2 ]] || exit 2; smoke_walltime="$2"; shift 2 ;;
        *) echo 'Usage: commit_deepswe_smoke.sh [--dry-run] [--walltime HH:MM:SS]' >&2; exit 2 ;;
    esac
done
[[ "$smoke_walltime" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo 'Invalid walltime' >&2; exit 2; }
args=(--job-name=deepswe-smoke-qwen35 --nodes=1 --ntasks=1 --cpus-per-task=8
      --mem=64G --gres="${DEEPSWE_GRES:-gpu:a100:1}" --time="$smoke_walltime"
      --export="ALL,DEEPSWE_SMOKE_WALLTIME=$smoke_walltime"
      --account=conf-cvpr-2026.11.23-ghanembs --chdir="$DEEPSWE_ROOT"
      --output=logs/deepswe_%j.out --error=logs/deepswe_%j.err)
[[ -z "${DEEPSWE_PARTITION:-}" ]] || args+=(--partition="$DEEPSWE_PARTITION")
if [[ "$dry_run" == 1 ]]; then
    printf '%q ' sbatch "${args[@]}" cluster/ibex/deepswe/job.sh
    printf '\n'
    exit 0
fi
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'OPENAI_API_KEY is required in .env.' >&2; exit 1; }
# Never queue training against an unvalidated candidate image.
python3 cluster/ibex/deepswe/check_full_ready.py
bash cluster/ibex/deepswe/prepare_uv.sh
bash cluster/ibex/deepswe/prepare_mini.sh
sbatch "${args[@]}" cluster/ibex/deepswe/job.sh
