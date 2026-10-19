"""Prepare pinned public assets. Never copy test tasks into a training home."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/eval/tasks/deepswe'))

import json
import os
import subprocess
import urllib.request
from full_suite import manifest, REVISION
from model_profiles import QWEN35_ID, QWEN35_REVISION


def main():
    from huggingface_hub import snapshot_download
    root = Path(__file__).resolve().parents[3]
    storage = Path(os.environ['DEEPSWE_STORAGE_IMAGES'])
    source = storage / 'deep-swe-source'
    if not source.exists():
        subprocess.run(['git', 'clone', 'https://github.com/datacurve-ai/deep-swe.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', REVISION], check=True)
    suite = manifest(source)
    (storage / 'deepswe_manifest.json').write_text(json.dumps(suite, indent=2) + '\n')
    snapshot = Path(snapshot_download(QWEN35_ID, revision=QWEN35_REVISION, max_workers=4))
    # Never use the submitted model's mutable chat template in the evaluator.
    template = snapshot / 'chat_template.jinja'
    (root / 'src/eval/templates/qwen3_5.jinja').write_bytes(template.read_bytes())
    reference = root / 'src/judges/judge_tools/reference_configs/Qwen_Qwen3.5-9B.json'
    reference.write_bytes((snapshot / 'config.json').read_bytes())
    (storage / 'qwen35_assets.json').write_text(json.dumps(dict(model=QWEN35_ID,
        revision=QWEN35_REVISION, snapshot=str(snapshot), dataset_revision=REVISION,
        task_count=len(suite['tasks'])), indent=2) + '\n')
    print('Prepared pinned model cache and all 113 host-only test tasks.')


if __name__ == '__main__':
    main()
