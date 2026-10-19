#!/bin/bash
# Submit on an Ibex compute node; never overwrite the active image here.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source cluster/ibex/deepswe/common.sh
build_dir="$DEEPSWE_STORAGE_RESULTS/image_build_${SLURM_JOB_ID}"
image_dir="$DEEPSWE_STORAGE_IMAGES/build_${SLURM_JOB_ID}"
mkdir -p "$image_dir"
mkdir -p "$build_dir/containers"
cp containers/standard.def containers/requirements-direct.txt "$build_dir/containers/"
# CPU build nodes cannot auto-detect the target GPU backend. Match the CUDA
# 12.9 base and the fixed eval SIF instead; preserve the original recipe too.
cp containers/standard.def "$build_dir/standard.original.def"
sed -i 's/--torch-backend=auto/--torch-backend=cu129/g' "$build_dir/containers/standard.def"
if [[ "${1:-}" == --from-eval ]]; then
    cp containers/deepswe_standard.def "$build_dir/containers/standard.def"
    # Keep the immutable input SIF alive if its default pathname is later changed.
    ln "$DEEPSWE_EVAL_SIF" "$image_dir/vllm_debug.sif"
    ln -s "$image_dir/vllm_debug.sif" "$build_dir/containers/vllm_debug.sif"
fi
export SINGULARITY_TMPDIR="$(mktemp -d /tmp/deepswe-build.XXXXXX)"
trap 'rm -rf -- "$SINGULARITY_TMPDIR"; rm -f -- "$image_dir/standard.sif.partial"' EXIT
unset APPTAINER_BIND SINGULARITY_BIND SINGULARITY_BINDPATH
cd "$build_dir"
sha256sum containers/* > sources.sha256
singularity build --fakeroot "$SINGULARITY_TMPDIR/standard.sif" containers/standard.def
cp "$SINGULARITY_TMPDIR/standard.sif" "$image_dir/standard.sif.partial"
mv "$image_dir/standard.sif.partial" "$image_dir/standard.sif"
ln -s "$image_dir/standard.sif" "$build_dir/standard.sif"
singularity inspect --deffile "$build_dir/standard.sif" > embedded.def
sha256sum standard.sif > image.sha256
echo "BUILT: $build_dir/standard.sif (validation required before promotion)"
