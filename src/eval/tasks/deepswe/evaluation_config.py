"""DeepSWE task contract and PostTrain generation defaults (not smoke limits)."""
from pathlib import Path
import re


# Re-export smoke settings for the shared task-container runtime.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.eval.tasks.deepswe_smoke.config import (
    TASK_ID, TASK_DIRECTORY, TEACHING_NAME, BASE_COMMIT, TASK_IMAGE, EXPERIMENT_MODE, PROTOCOL,
)

HARNESS = 'mini-swe-agent'
MINI_REVISION = '04d809ceab9df28f9adaed044884180159172930'
PIER_REFERENCE_REVISION = '4d3c14041d16443f3f9f460dcdf23629994a304e'


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
            'gpu_memory_utilization': 0.3, 'max_tokens': 4000, 'temperature': None, 'max_model_len': 32768,
            'max_turns': None, 'history_truncation': False, 'summarization': False,
            'harness': HARNESS, 'mini_revision': MINI_REVISION,
            'pier_reference_revision': PIER_REFERENCE_REVISION}
