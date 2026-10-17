"""Run inside the existing eval SIF; verifies the real Inspect API with a mock model."""
import asyncio
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, '/harness')
from inspect_eval import shell_loop, verifier, protocol_message, PROTOCOL
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model, ModelOutput

model = get_model('mockllm/model', custom_outputs=[ModelOutput.from_content('mockllm/model', text) for text in ['{"command":"git status"}', '{"done":true}']])
with patch('inspect_eval.rpc', side_effect=[{'exit_code':0, 'output':'clean'}, {'reward':0}]):
    logs = inspect_eval(Task(dataset=[Sample(input='Fix this repository')],
        solver=[protocol_message(), shell_loop()], scorer=verifier()), model=model, log_dir='/output/logs', display='none')
assert logs[0].samples[0].messages[0].content == PROTOCOL
assert logs[0].status == 'success', logs[0].error
assert json.loads(Path('/output/reward.json').read_text()) == {'reward': 0}
print('Inspect mock-model shell loop and verifier passed')
