#!/bin/bash
set -euo pipefail
source src/deepswe/common.sh
export DEEPSWE_CONTAINER_TMPDIR="$(mktemp -d /tmp/deepswe-framework.XXXXXX)"
trap 'rm -rf -- "$DEEPSWE_CONTAINER_TMPDIR"' EXIT
python3 - <<'PY'
import os,sys
from pathlib import Path
sys.path.insert(0,'src/deepswe')
from runtime import ROOT, container, run
output=Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('framework_check_' + os.environ['SLURM_JOB_ID'])
output.mkdir()
cmd=container(Path(os.environ['DEEPSWE_EVAL_SIF']),output/'home',[
    f'{ROOT / "src/deepswe"}:/harness:ro',
    f'{ROOT / "src/eval/per_sample_seed.py"}:/harness_seed.py:ro',
    f'{ROOT / "dev_utils/deepswe/test_inspect.py"}:/test_inspect.py:ro',
    f'{output}:/output'])
code,out=run(cmd+['python','/test_inspect.py'],timeout=600)
print(out,flush=True)
(output/'inspect_check.txt').write_text(out)
PY
