#!/bin/bash
# Run as a separate CPU batch job; image preparation is outside the 30m GPU test.
set -euo pipefail
# sbatch copies this script into its spool; --chdir sets the repository root.
source src/deepswe/common.sh
mkdir -p "$(dirname "$DEEPSWE_TASK_SIF")"
if [[ ! -f "$DEEPSWE_TASK_SIF" ]]; then
    singularity pull "$DEEPSWE_TASK_SIF" docker://public.ecr.aws/d3j8x8q7/swe-bench-202605:kh7191qb52n5pfwh0a4yhahmt18343sn-v1.1
fi
for path in "$DEEPSWE_TRAIN_SIF" "$DEEPSWE_EVAL_SIF"; do
    [[ -f "$path" ]] || { echo "Missing ML image: $path; build the corresponding .def using singularity build on a suitable build host." >&2; exit 1; }
done
# Fail closed if this installation cannot isolate task networking.
singularity exec --containall --cleanenv --no-mount hostfs --net --network none "$DEEPSWE_TASK_SIF" \
    bash -c 'test -d /app/.git && python -c "import pytest, fastapi"'
