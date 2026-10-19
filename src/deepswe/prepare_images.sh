#!/bin/bash
# Run as a separate job; image preparation is outside the 90m training budget.
# The conference account requires a GPU allocation even for image preparation.
set -euo pipefail
# sbatch copies this script into its spool; --chdir sets the repository root.
source src/deepswe/common.sh
export SINGULARITY_TMPDIR="$(mktemp -d /tmp/deepswe-images.XXXXXX)"
trap 'rm -rf -- "$SINGULARITY_TMPDIR"' EXIT
mkdir -p "$(dirname "$DEEPSWE_TASK_SIF")"
if [[ ! -f "$DEEPSWE_TASK_SIF" ]]; then
    singularity pull "$DEEPSWE_TASK_SIF" "docker://$(PYTHONPATH=src/deepswe python3 -c 'from evaluation_config import TASK_IMAGE; print(TASK_IMAGE)')"
fi
for path in "$DEEPSWE_TRAIN_SIF" "$DEEPSWE_EVAL_SIF"; do
    [[ -f "$path" ]] || { echo "Missing ML image: $path; build the corresponding .def using singularity build on a suitable build host." >&2; exit 1; }
done
# Fail closed if this installation cannot isolate task networking.
singularity exec --fakeroot --containall --cleanenv --no-mount hostfs --net --network none "$DEEPSWE_TASK_SIF" \
    bash -c 'test -d /app/.git && node --version && python3 --version'
