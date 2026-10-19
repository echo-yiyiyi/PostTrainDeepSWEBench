"""DeepSWE task contract and PostTrain generation defaults (not smoke limits)."""
from pathlib import Path
import re


TASK_ID = 'happy-dom-abort-pending-body-reads'
TASK_DIRECTORY = 'happy_dom_task'
TEACHING_NAME = 'happy_dom'
BASE_COMMIT = '82a0888cb2c87a6123e05424b528f8e8c9b3e426'
TASK_IMAGE = 'public.ecr.aws/d3j8x8q7/swe-bench-202605:kh7c2re7cvbseq7xz6samd1xr182y1dc-v1.1'
EXPERIMENT_MODE = 'task_specific_training'
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


PROTOCOL = '''Evaluation uses the official mini-swe-agent v2 agent and mini.yaml
prompt templates. Read mini_swe_config.yaml for the exact system/instance and
observation/error templates, and mini_swe_bash_tool.json for the tool schema.
The model emits native bash function calls, NOT {"command":...}/{"done":true}
standalone JSON actions. Use templates/qwen3.jinja with tools=[the bash schema]
when preparing training inputs. Assistant tool calls use <tool_call> blocks;
tool observations are rendered with <tool_response> by the same chat template.
Commands run in /app in independent shells; repository edits persist, shell
variables and cd do not. Commit all changes before submitting. Submit via the
bash tool: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT (alone).
max_tokens controls output per API request via model.model_kwargs.max_tokens.
The server overrides only generation_config.max_new_tokens with this budget,
so a checkpoint cannot silently impose a smaller output limit.
The server's total input-plus-output context is max_model_len. Native mini-swe
does not automatically summarize; context exhaustion ends the rollout, and the
existing committed diff is graded by the official separate verifier.
'''
