"""Execute inside the same writable home and cache mounts as the training agent."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText
from transformers import Qwen3_5ForConditionalGeneration

task = Path('/home/ben/task')
assert Path.cwd() == task
assert sys.prefix == '/home/ben/.venv'
required_files = ('evaluate.py', 'timer.sh', 'model_profile.json',
             'templates/qwen3_5.jinja', 'mini_swe_bash_tool.json',
             'mini_swe_config.yaml', 'evaluation_config.json')
file_checks = {}
for name in required_files:
    path = task / name
    file_checks[name] = path.is_file() and os.access(path, os.R_OK)
    assert file_checks[name], name
profile = json.loads((task / 'model_profile.json').read_text())
assert profile['enable_thinking'] is True
source = Path(profile['cached_model'])
assert source.name == profile['revision']
assert source.is_dir(), source
index = source / 'model.safetensors.index.json'
if index.is_file():
    weights = sorted(set(json.loads(index.read_text())['weight_map'].values()))
else:
    weights = ['model.safetensors']
for name in weights:
    path = source / name
    assert path.is_file() and os.access(path, os.R_OK), path
config = AutoConfig.from_pretrained(source, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
schema = json.loads((task / 'mini_swe_bash_tool.json').read_text())
assert schema['function']['name'] == 'bash'
template = (task / 'templates/qwen3_5.jinja').read_text()
assert template == (source / 'chat_template.jinja').read_text()
rendered = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'Run printf hello using bash.'}],
    tools=[schema], chat_template=template, tokenize=False,
    add_generation_prompt=True, enable_thinking=True)
assert rendered.rstrip().endswith('<think>'), rendered[-200:]
assert os.access(task, os.W_OK)
cpu_only = '--cpu-only' in sys.argv
commands = [['bash', 'timer.sh'], ['codex', '--version'], ['git', '--version'],
            ['python', 'evaluate.py', '--help']]
if not cpu_only:
    commands.append(['nvidia-smi', '-L'])
for command in commands:
    subprocess.run(command, check=True)

# Exercise a real offline install, not just --help or a mocked package manager.
wheel = task / 'prompt_probe-0.0.0-py3-none-any.whl'
with zipfile.ZipFile(wheel, 'w') as z:
    z.writestr('prompt_probe.py', 'AVAILABLE = True\n')
    z.writestr('prompt_probe-0.0.0.dist-info/METADATA',
               'Metadata-Version: 2.1\nName: prompt-probe\nVersion: 0.0.0\n')
    z.writestr('prompt_probe-0.0.0.dist-info/WHEEL',
               'Wheel-Version: 1.0\nGenerator: probe\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
    z.writestr('prompt_probe-0.0.0.dist-info/RECORD', '')
subprocess.run(['uv', 'pip', 'install', '--python', sys.executable,
                '--no-index', str(wheel)], check=True)
importlib.invalidate_caches()
assert importlib.import_module('prompt_probe').AVAILABLE
(task / 'external_api_usage.jsonl').touch(exist_ok=False)
assert not (Path('/home/ben') / 'test_data.json').exists()
assert not (task / 'reference_materials').exists()
(task / 'prompt_environment.json').write_text(json.dumps({
    'passed': True, 'enable_thinking': True, 'model_type': config.model_type,
    'cached_model_readable': True, 'offline_package_install': True,
    'template_render': True, 'files_present': True,
    'required_files': file_checks, 'cached_weight_files': weights,
    'output_paths': {
        'final_model/': 'created by training; not required at startup',
        'trial_metrics.json': 'created by evaluation; not required at startup',
        'external_api_usage.jsonl': 'creation verified by this probe',
    },
    'cli_commands': True, 'held_out_files_absent': True,
    'gpu_check': 'skipped (CPU-only probe)' if cpu_only else 'passed',
}, indent=2))
