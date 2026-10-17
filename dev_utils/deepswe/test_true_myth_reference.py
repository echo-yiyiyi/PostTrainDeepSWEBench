"""Positive control for the selected task adapter; no training/model API calls."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, 'src/deepswe')
from runtime import ROOT, TaskEnvironment, container, run
from evaluation_config import TASK_ID

assert TASK_ID == 'true-myth-iterable-collection-combinators'
output = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('true_myth_reference_check_' + os.environ['SLURM_JOB_ID'])
output.mkdir()
task = TaskEnvironment(output, Path(os.environ['DEEPSWE_TASK_SIF']))
# This is explicitly an oracle adapter test, not an evaluated model's submission.
solution = ROOT / 'src/deepswe/teaching_materials/true_myth/solution'
run(container(task.image, task.home, [f'{task.app}:/app', f'{solution}:/solution:ro'], pwd='/app')
    + ['bash', '/solution/solve.sh'])
reward = task.verify()
(output/'check_metadata.json').write_text(json.dumps({'oracle_control': True, 'model_evaluation': False, 'task_id': TASK_ID}, indent=2))
assert reward['reward'] == 1, reward
print('PASS: official true-myth reference solution passes the unmodified verifier:', reward, flush=True)
