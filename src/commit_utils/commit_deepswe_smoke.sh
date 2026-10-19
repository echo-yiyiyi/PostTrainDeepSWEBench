#!/bin/bash
# One outer Codex post-training run, one exposed training task, one evaluation seed.
set -euo pipefail
source "$(dirname "$0")/../../cluster/ibex/deepswe/common.sh"
mkdir -p logs
args=(--job-name=posttrain-deepswe --nodes=1 --ntasks=1 --cpus-per-task=8
      --mem=64G --gres="${DEEPSWE_GRES:-gpu:a100:1}" --time="${DEEPSWE_JOB_WALLTIME:-01:30:00}"
      --chdir="$DEEPSWE_ROOT"
      --output=logs/deepswe_%j.out --error=logs/deepswe_%j.err)
[[ -z "${DEEPSWE_PARTITION:-}" ]] || args+=(--partition="$DEEPSWE_PARTITION")
account_args=()
[[ -z "${DEEPSWE_ACCOUNT:-}" ]] || account_args+=(--account="$DEEPSWE_ACCOUNT")
args+=("${account_args[@]}")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' sbatch "${args[@]}" cluster/ibex/deepswe/job.sh
    printf '\n'
    [[ -f "$DEEPSWE_TASK_SIF" ]] || echo 'A separate image preparation job will precede this job.'
    exit 0
fi
for path in "$DEEPSWE_TRAIN_SIF" "$DEEPSWE_EVAL_SIF"; do
    [[ -f "$path" ]] || { echo "Missing image: $path. See cluster/ibex/deepswe/README.md." >&2; exit 1; }
done
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'OPENAI_API_KEY is required in .env (same as agent=codex).' >&2; exit 1; }
bash cluster/ibex/deepswe/prepare_uv.sh
bash cluster/ibex/deepswe/prepare_mini.sh
if [[ ! -f "$DEEPSWE_TASK_SIF" ]]; then
    prepare_id="${DEEPSWE_PREPARE_JOB_ID:-}"
    if [[ -z "$prepare_id" ]]; then
        prepare_id=$(sbatch --parsable "${account_args[@]}" --job-name=deepswe-prepare --nodes=1 --ntasks=1 \
            --cpus-per-task=4 --mem=16G --gres="${DEEPSWE_GRES:-gpu:a100:1}" --time=00:30:00 --chdir="$DEEPSWE_ROOT" \
            --output=logs/deepswe_prepare_%j.out --error=logs/deepswe_prepare_%j.err \
            cluster/ibex/deepswe/prepare_images.sh)
        prepare_id="${prepare_id%%;*}"
        echo "Image preparation job: $prepare_id"
    fi
    [[ "$prepare_id" =~ ^[0-9]+$ ]] || { echo 'Invalid preparation job ID' >&2; exit 1; }
    args+=(--dependency="afterok:$prepare_id" --kill-on-invalid-dep=yes)
fi
sbatch "${args[@]}" cluster/ibex/deepswe/job.sh
