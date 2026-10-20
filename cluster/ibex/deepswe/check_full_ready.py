"""Fail before a paid training job if preparation artifacts are incomplete."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/eval/tasks/deepswe'))

import json
import os
from full_suite import load_manifest, verify_task
from model_profiles import QWEN35_ID, QWEN35_REVISION

storage = Path(os.environ['DEEPSWE_STORAGE_IMAGES'])
if not (storage / 'qwen35_v032.sif').is_file():
    raise FileNotFoundError(storage / 'qwen35_v032.sif')
ready = json.loads((storage / 'qwen35_v032_ready.json').read_text())
if (ready.get('model') != QWEN35_ID or ready.get('revision') != QWEN35_REVISION
        or not ready.get('training_export') or not ready.get('native_tool_call')
        or not ready.get('end_to_end_evaluation') or ready.get('max_model_len') != 32768
        or ready.get('output_budget') != 4000 or ready.get('enable_thinking') is not True
        or not ready.get('prompt_environment') or not ready.get('evaluation_client')):
    raise ValueError('Qwen3.5 image validation is missing or mismatched')
if ready.get('versions', {}).get('vllm') not in ('0.32.0', '0.32.0+cu129') or ready.get('versions', {}).get('transformers') != '5.19.0':
    raise ValueError('The upgraded training/evaluation stack has not passed its GPU gate')
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
