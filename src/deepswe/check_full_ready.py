"""Fail before a paid training job if preparation artifacts are incomplete."""
import json
import os
from pathlib import Path
from full_suite import load_manifest, verify_task
from model_profiles import QWEN35_ID, QWEN35_REVISION

storage = Path(os.environ['DEEPSWE_STORAGE_IMAGES'])
ready = json.loads((storage / 'qwen35_ready.json').read_text())
if (ready.get('model') != QWEN35_ID or ready.get('revision') != QWEN35_REVISION
        or not ready.get('training_export') or not ready.get('native_tool_call')):
    raise ValueError('Qwen3.5 image validation is missing or mismatched')
suite = load_manifest(storage / 'deepswe_manifest.json')
for task in suite['tasks']:
    verify_task(task)
assets = json.loads((storage / 'qwen35_assets.json').read_text())
if assets['revision'] != QWEN35_REVISION:
    raise ValueError('Model cache revision mismatch')
snapshot = Path(assets['snapshot'])
index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    if not (snapshot / name).is_file():
        raise FileNotFoundError(snapshot / name)
print('Prepared model cache, image gate and all 113 held-out tasks are ready.')
