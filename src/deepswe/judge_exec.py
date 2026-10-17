"""Execute the shared judge command in an isolated Ibex Singularity home."""
import os
from pathlib import Path
import sys
import subprocess
from runtime import container, run

home, output = map(Path, sys.argv[1:])
env = {'CODEX_API_KEY': os.environ['CODEX_API_KEY'],
       'OPENAI_API_KEY': os.environ['CODEX_API_KEY'],
       'PYTHONNOUSERSITE': '1', 'CODEX_HOME': '/home/ben/.codex',
       'JUDGE_MODEL': os.environ['DEEPSWE_JUDGE_MODEL'],
       'JUDGE_REASONING_EFFORT': os.environ['DEEPSWE_JUDGE_EFFORT'],
       'JUDGE_CODEX_VERSION': os.environ['DEEPSWE_JUDGE_CLI']}
# Judge command construction is shared with original PostTrain, not duplicated.
from runtime import ROOT
script = '''set -euo pipefail
source /framework/src/judges/judge_lib.sh
# Loading the library supplies defaults but does not load a judge config.
pin_prefix="/home/ben/.codex-cli-${JUDGE_CODEX_VERSION}"
codex_bin=codex
if [[ -n "$JUDGE_CODEX_VERSION" ]]; then
    if [[ ! -x "$pin_prefix/bin/codex" ]]; then
        npm install -g --prefix "$pin_prefix" --no-fund --no-audit "@openai/codex@$JUDGE_CODEX_VERSION"
    fi
    codex_bin="$pin_prefix/bin/codex"
fi
judge_codex_args "$(cat /home/ben/judge_prompt.txt)"
exec "$codex_bin" "${JUDGE_CODEX_ARGS[@]}" < /dev/null
'''
try:
    with output.open('w') as log:
        code, _ = run(container(Path(os.environ['DEEPSWE_TRAIN_SIF']), home,
            [f'{ROOT / "src"}:/framework/src:ro'], network=True, pwd='/home/ben/task', env=env)
            + ['bash', '-c', script], timeout=int(os.environ.get('JUDGE_TIMEOUT_SEC', '7200')),
            check=False, stdout=log)
except subprocess.TimeoutExpired:
    code = 124
# Shared orchestration warns on absent verdicts and continues to final evaluation.
print('Judge exit code:', code)
