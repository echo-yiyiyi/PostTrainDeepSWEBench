"""GPU preparation gate: actual 9B training/export and vLLM native tool round trip."""
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/deepswe'))
from runtime import container, run, clean_env
from model_profiles import serving_args, QWEN35_ID, QWEN35_REVISION

image = Path(sys.argv[1]).resolve()
assets = json.loads((Path(os.environ['DEEPSWE_STORAGE_IMAGES']) / 'qwen35_assets.json').read_text())
report = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('qwen35_check_' + os.environ['SLURM_JOB_ID'])
report.mkdir(exist_ok=False)
with tempfile.TemporaryDirectory(dir=os.environ['DEEPSWE_CONTAINER_TMPDIR']) as tmp:
    tmp = Path(tmp)
    out = tmp / 'output'
    out.mkdir()
    with (report / 'train.log').open('w') as log:
        run(container(image, tmp / 'train_home', [f'{Path(assets["snapshot"]).parent.parent}:/hf_model:ro',
            f'{out}:/output', f'{ROOT / "dev_utils/deepswe/qwen35_probe.py"}:/probe.py:ro',
            f'{ROOT / "src/deepswe/setup_training_env.sh"}:/setup.sh:ro'], gpu=True, env={'MODEL_SOURCE': '/hf_model/snapshots/' + QWEN35_REVISION}) +
            ['bash', '-c', 'bash /setup.sh && /home/ben/.venv/bin/python /probe.py'], timeout=1800, stdout=log)
    (report / 'train_probe.json').write_bytes((out / 'train_probe.json').read_bytes())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(24)
    cmd = container(image, tmp / 'eval_home', [f'{out / "checkpoint"}:/model:ro',
        f'{ROOT / "src/eval/templates"}:/templates:ro'], gpu=True, network=True,
        env={'VLLM_API_KEY': key, 'HF_HUB_OFFLINE': '1'})
    cmd += ['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', '/model',
        '--host', '127.0.0.1', '--port', str(port), '--served-model-name', 'probe',
        '--max-model-len', '8192', '--gpu-memory-utilization', '0.8',
        '--enable-auto-tool-choice', '--override-generation-config', '{"max_new_tokens":4000}']
    cmd += serving_args(QWEN35_ID, ROOT / 'src/eval/templates')
    with (report / 'vllm.log').open('w') as log:
        proc = subprocess.Popen(cmd, env=clean_env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('vLLM startup failed')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('vLLM readiness')
                    time.sleep(2)
            schema = json.loads((ROOT / 'src/deepswe/mini_swe/bash_tool.json').read_text())
            body = {'model':'probe', 'messages':[{'role':'user','content':'Use the bash tool once to run: printf hello'}],
                    'tools':[schema], 'max_tokens':4000, 'temperature':0}
            req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                headers={'Content-Type':'application/json', 'Authorization':'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.load(response)
            (report / 'tool_response.json').write_text(json.dumps(result, indent=2))
            calls = result['choices'][0]['message'].get('tool_calls')
            if not calls or calls[0]['function']['name'] != 'bash':
                raise RuntimeError('No native bash tool call')
            args = json.loads(calls[0]['function']['arguments'])
            assert isinstance(args['command'], str)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
(report / 'ready.json').write_text(json.dumps(dict(model=QWEN35_ID, revision=QWEN35_REVISION,
    image=str(image), training_export=True, native_tool_call=True, output_budget=4000), indent=2))
print('Qwen3.5 real-model training/export and vLLM tool-call checks passed.')
