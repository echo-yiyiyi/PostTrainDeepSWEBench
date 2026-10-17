#!/bin/bash
# Runs inside the training SIF as the submitting user, before Codex starts.
set -euo pipefail
export VIRTUAL_ENV=/home/ben/.venv
export PATH="$VIRTUAL_ENV/bin:/home/ben/.local/bin:$PATH"
export UV_PYTHON_DOWNLOADS=never
export UV_CONSTRAINT=/home/ben/training-constraints.txt
uv --version
uv venv --python /usr/bin/python --system-site-packages "$VIRTUAL_ENV"
python - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import torch
import transformers
import accelerate
import peft
import datasets
import safetensors

assert sys.prefix == '/home/ben/.venv', sys.executable
assert os.access(sys.prefix, os.W_OK), 'Training virtual environment is not writable'
if os.environ.get('DEEPSWE_REQUIRE_CUDA') == '1' and not torch.cuda.is_available():
    raise RuntimeError('Training GPU is unavailable')
packages = ('torch', 'transformers', 'accelerate', 'peft', 'datasets',
            'safetensors', 'tokenizers', 'numpy')
versions = {name: importlib.metadata.version(name) for name in packages}
Path(os.environ['UV_CONSTRAINT']).write_text(''.join(
    f'{name}=={version}\n' for name, version in versions.items()))
print(json.dumps({'python': sys.executable, 'uid': os.getuid(),
                  'cuda_available': torch.cuda.is_available(), 'packages': versions}))
PY
