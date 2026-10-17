"""DeepSWE task contract and PostTrain generation defaults (not smoke limits)."""
from pathlib import Path
import re


TASK_ID = 'true-myth-iterable-collection-combinators'
TASK_DIRECTORY = 'true_myth_task'
BASE_COMMIT = 'd8fbebc75de4991a32354518beff1abf628d0b07'
TASK_IMAGE = 'public.ecr.aws/d3j8x8q7/swe-bench-202605:kh74r2t7kdnt7h2efdk0hf5asx82zr0s-v1.1'
EXPERIMENT_MODE = 'task_specific_training'


def task_timeout(path, section):
    text = Path(path).read_text()
    block = re.search(r'^\[' + section + r'\]\s*\n(.*?)(?=^\[|\Z)', text, re.M | re.S)
    if not block:
        raise ValueError(f'Missing [{section}] in task contract')
    value = re.search(r'^timeout_sec\s*=\s*([0-9.]+)', block[1], re.M)
    if not value:
        raise ValueError(f'Missing timeout_sec in [{section}]')
    return int(float(value[1]))


def defaults(task_file):
    return {'agent_timeout_seconds': task_timeout(task_file, 'agent'),
            'verifier_timeout_seconds': task_timeout(task_file, 'verifier'),
            'gpu_memory_utilization': 0.3, 'max_tokens': 4000, 'temperature': None, 'max_model_len': None,
            'max_turns': None, 'history_truncation': False}


PROTOCOL = '''You are solving a software engineering task in /app.
Respond with exactly one JSON object per turn: {"command": "a bash command"}.
Commands execute in /app in the task environment; edits to /app persist.
Inspect and edit the repository and test your changes. Before finishing, commit
all your changes with git (configure user.name/user.email locally if needed).
Only committed changes are graded. When finished respond {"done": true}.
Do not print markdown fences.'''
