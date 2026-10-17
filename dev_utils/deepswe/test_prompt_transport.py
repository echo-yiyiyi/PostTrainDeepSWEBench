"""Run on a host with Singularity loaded. No GPU, network or model API needed."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='prompt-check-', dir=ROOT / 'dev_utils/deepswe') as tmp:
    home = Path(tmp)
    prompt = ('First line\nDeepSWE single task; 12 minutes.\n'
              'Literal $HOME, $(printf unexpected), "quotes", and \'quotes\'.\n'
              'Save final_model/config.json')
    (home / 'training_prompt.txt').write_text(prompt)
    shutil.copy2(ROOT / 'src/deepswe/start_agent.sh', home / 'start_agent.sh')
    (home / 'agent_solve.sh').write_text(
        "#!/bin/bash\npython -c 'import json, os; print(json.dumps(os.environ[\"PROMPT\"]))'\n")
    result = subprocess.run([
        'singularity', 'exec', '--containall', '--cleanenv', '--net', '--network', 'none',
        '--home', f'{home}:/home/ben', '--pwd', '/home/ben',
        str(ROOT / 'containers/standard.sif'), 'bash', '/home/ben/start_agent.sh',
    ], capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(result.stdout) == prompt, result.stdout
    print('PASS: full multiline prompt and literal shell characters preserved at agent entry.')
