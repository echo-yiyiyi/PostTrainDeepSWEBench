#!/bin/bash
# Training -> independent all-test array -> complete-only aggregate.
set -euo pipefail
source "$(dirname "$0")/../../cluster/ibex/deepswe/common.sh"
account=conf-cvpr-2026.11.23-ghanembs
common=(--account="$account" --nodes=1 --ntasks=1
        --cpus-per-task=8 --mem=64G --chdir="$DEEPSWE_ROOT")
[[ -z "${DEEPSWE_PARTITION:-}" ]] || common+=(--partition="$DEEPSWE_PARTITION")
if [[ "${1:-}" == --dry-run ]]; then
    echo '10h training + judges, followed by all 113 test tasks (array concurrency 4), then complete-only aggregation.'
    echo "Model: Qwen/Qwen3.5-9B; results: $DEEPSWE_STORAGE_RESULTS/deepswe; account: $account"
    exit 0
fi
dependency=()
if [[ "${1:-}" == --after-validation ]]; then
    [[ $# -eq 2 && "$2" =~ ^[0-9]+$ ]] || { echo 'Expected a validation job ID' >&2; exit 2; }
    dependency=(--dependency="afterok:$2" --kill-on-invalid-dep=yes)
    shift 2
fi
[[ $# -eq 0 ]] || { echo 'Usage: commit_deepswe.sh [--dry-run | --after-validation JOB_ID]' >&2; exit 1; }
for file in "$DEEPSWE_STORAGE_IMAGES/qwen35_v032.sif" "$DEEPSWE_STORAGE_IMAGES/deepswe_manifest.json" "$DEEPSWE_STORAGE_IMAGES/qwen35_v032_ready.json"; do
    [[ -f "$file" ]] || { echo "Missing preparation/validation artifact: $file" >&2; exit 1; }
done
if [[ ${#dependency[@]} -eq 0 ]]; then
    python3 cluster/ibex/deepswe/check_full_ready.py
fi
bash cluster/ibex/deepswe/prepare_uv.sh
bash cluster/ibex/deepswe/prepare_mini.sh
mkdir -p logs

[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'OPENAI_API_KEY is required' >&2; exit 1; }
train=$(sbatch --parsable "${common[@]}" --gres="${DEEPSWE_GRES:-gpu:a100:1}" "${dependency[@]}" --job-name=deepswe-train --time="${DEEPSWE_FULL_WALLTIME:-12:00:00}" \
    --output=logs/deepswe_train_%j.out --error=logs/deepswe_train_%j.err cluster/ibex/deepswe/full_job.sh train)
train=${train%%;*}
result="$DEEPSWE_STORAGE_RESULTS/deepswe/$train"
eval_job=$(sbatch --parsable "${common[@]}" --gres="${DEEPSWE_EVAL_GRES:-gpu:a100:1}" --job-name=deepswe-eval --time=05:00:00 \
    --array="0-112%${DEEPSWE_FULL_EVAL_CONCURRENCY:-4}" --dependency="afterok:$train" --kill-on-invalid-dep=yes \
    --output=logs/deepswe_eval_%A_%a.out --error=logs/deepswe_eval_%A_%a.err cluster/ibex/deepswe/full_job.sh evaluate "$result")
eval_job=${eval_job%%;*}
aggregate=$(sbatch --parsable --account=pi-ghanembs --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
    --chdir="$DEEPSWE_ROOT" --job-name=deepswe-aggregate --time=00:10:00 \
    --dependency="afterok:$eval_job" --kill-on-invalid-dep=yes \
    --output=logs/deepswe_aggregate_%j.out --error=logs/deepswe_aggregate_%j.err cluster/ibex/deepswe/full_job.sh aggregate "$result")
printf 'Training: %s\nFinal evaluation: %s\nAggregate: %s\nResults: %s\n' "$train" "$eval_job" "$aggregate" "$result"
