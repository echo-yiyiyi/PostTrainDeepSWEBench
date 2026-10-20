"""Trusted model profiles; checkpoint metadata cannot choose the eval protocol."""
from pathlib import Path
import json

QWEN35_ID = 'Qwen/Qwen3.5-9B'
QWEN35_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'


def serving_args(model_id: str, templates: Path, *, enable_thinking: bool = True) -> list[str]:
    if model_id == QWEN35_ID:
        template = templates / 'qwen3_5.jinja'
        if not template.is_file():
            raise FileNotFoundError('Prepare the pinned Qwen3.5 template first')
        return ['--chat-template', '/templates/qwen3_5.jinja',
                '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
                '--language-model-only', '--default-chat-template-kwargs', json.dumps({'enable_thinking': enable_thinking})]
    if model_id.startswith('Qwen/Qwen3-'):
        return ['--chat-template', '/templates/qwen3.jinja', '--tool-call-parser', 'hermes']
    raise ValueError(f'Unsupported model profile: {model_id}')
