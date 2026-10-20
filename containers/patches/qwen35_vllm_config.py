"""vLLM 0.17.1 compatibility with Transformers 5.2 RoPE validation.

vLLM passes a list of ignored keys; Transformers unions these keys with a set.
Only normalize the collection type. Do not change any model/RoPE parameters.
"""
from pathlib import Path
import importlib.util
import re


def patch_source(source: str) -> str:
    pattern = r'(kwargs\["ignore_keys_at_rope_validation"\] = )\[(\s*"mrope_section",\s*"mrope_interleaved",?\s*)\]'
    updated, count = re.subn(pattern, r'\1{\2}', source)
    if count != 1:
        raise RuntimeError(f'Expected one vLLM 0.17.1 RoPE compatibility site, found {count}')
    return updated


if __name__ == '__main__':
    spec = importlib.util.find_spec('vllm.transformers_utils.configs.qwen3_5')
    if spec is None or spec.origin is None:
        raise RuntimeError('Missing vLLM Qwen3.5 config implementation')
    path = Path(spec.origin)
    path.write_text(patch_source(path.read_text()))
