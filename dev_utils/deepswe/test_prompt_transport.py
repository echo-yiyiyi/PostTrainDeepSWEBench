"""Run on a compute node with fakeroot configured. No GPU/network/model API needed."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval/tasks/deepswe'))
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from runtime import container, clean_env
with tempfile.TemporaryDirectory(prefix='prompt-check-', dir=ROOT / 'dev_utils/deepswe') as tmp:
    home = Path(tmp)
    prompt = ('First line\nDeepSWE single task; 12 minutes.\n'
              'Literal $HOME, $(printf unexpected), "quotes", and \'quotes\'.\n'
              'Save final_model/config.json')
    (home / 'training_prompt.txt').write_text(prompt)
    shutil.copy2(ROOT / 'cluster/ibex/deepswe/start_agent.sh', home / 'start_agent.sh')
    (home / 'agent_solve.sh').write_text(
        "#!/bin/bash\npython -c 'import json, os; print(json.dumps(os.environ[\"PROMPT\"]))'\n")
    result = subprocess.run(container(ROOT / 'containers/standard.sif', home, pwd='/home/ben') +
        ['bash', '/home/ben/start_agent.sh'], env=clean_env(),
        capture_output=True, text=True, check=True, timeout=180)
    assert json.loads(result.stdout) == prompt, result.stdout
    print('PASS: full multiline prompt and literal shell characters preserved at agent entry.')
