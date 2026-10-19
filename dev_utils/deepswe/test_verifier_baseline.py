"""Verify an unchanged repository using the official task grader, without a model."""
import os
from pathlib import Path
import sys
sys.path.insert(0, 'src/eval/tasks/deepswe')
from runtime import TaskEnvironment
output = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('verifier_check_' + os.environ['SLURM_JOB_ID'])
output.mkdir()
task = TaskEnvironment(output, Path(os.environ['DEEPSWE_TASK_SIF']))
reward = task.verify()
assert (output / 'artifacts/model.patch').read_bytes() == b''
assert not reward.get('apply_failed'), reward
print('PASS: empty committed diff is uncontaminated; official baseline verifier ran:', reward, flush=True)
