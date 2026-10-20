"""GPU preparation gate: actual 9B training/export and vLLM native tool round trip."""
import json
import os
from pathlib import Path
import secrets
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval/tasks/deepswe'))
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from runtime import container, run, clean_env, cache_overlay
from model_profiles import serving_args, QWEN35_ID, QWEN35_REVISION

image = Path(sys.argv[1]).resolve()
assets = json.loads((Path(os.environ['DEEPSWE_STORAGE_IMAGES']) / 'qwen35_assets.json').read_text())
report = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('qwen35_check_' + os.environ['SLURM_JOB_ID'])
report.mkdir(exist_ok=False)
with tempfile.TemporaryDirectory(dir=os.environ['DEEPSWE_CONTAINER_TMPDIR']) as tmp:
    tmp = Path(tmp)
    out = tmp / 'output'
    out.mkdir()
    # Import the full serving entry point before spending time training.
    # Importing only `vllm` does not load optional media/native libraries.
    with (report / 'server_preflight.log').open('w') as log:
        run(container(image, tmp / 'preflight_home', gpu=True) +
            ['python', '-m', 'vllm.entrypoints.openai.api_server', '--help'],
            timeout=300, stdout=log)
    with (report / 'train.log').open('w') as log:
        run(container(image, tmp / 'train_home', [f'{Path(assets["snapshot"]).parent.parent}:/hf_model:ro',
            f'{out}:/output', f'{ROOT / "dev_utils/deepswe/qwen35_probe.py"}:/probe.py:ro',
            f'{ROOT / "cluster/ibex/deepswe/setup_training_env.sh"}:/setup.sh:ro'], gpu=True, env={'MODEL_SOURCE': '/hf_model/snapshots/' + QWEN35_REVISION}) +
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
        '--max-model-len', '32768', '--gpu-memory-utilization', '0.8',
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
            schema = json.loads((ROOT / 'src/eval/tasks/deepswe/mini_swe/bash_tool.json').read_text())
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
    # Exercise the real mini-swe rollout and separate verifier, using the same
    # context, model serving arguments, task contract and scoring path as smoke.
    from full_suite import load_manifest
    from trial_bridge import TrialBridge
    from run import prepare_training_home, prepare_protocol_files, TRAIN_ENV
    from training_support import make_general_prompt
    suite = load_manifest(Path(os.environ['DEEPSWE_STORAGE_IMAGES']) / 'deepswe_manifest.json')
    probe_run = tmp / 'probe_run'
    probe_home = probe_run / 'training_home'
    (probe_home / 'task').mkdir(parents=True)
    shutil.move(str(out / 'checkpoint'), str(probe_home / 'task/final_model'))
    (probe_run / 'run_config.json').write_text(json.dumps(dict(
        model_to_train=QWEN35_ID, harness='mini-swe-agent', experiment_mode='full_held_out', enable_thinking=True)))
    prepare_training_home(probe_home, modern=True)
    prepare_protocol_files(probe_home, QWEN35_ID)
    task = probe_home / 'task'
    shutil.copy2(ROOT / 'src/eval/tasks/deepswe/evaluate.py', task / 'evaluate.py')
    (task / 'timer.sh').write_text(
        '#!/bin/bash\n'
        f'remaining=$(({int(time.time()) + 36000} - $(date +%s)))\n'
        'printf "Remaining time (minutes:seconds): %d:%02d\\n" '
        '"$((remaining / 60))" "$((remaining % 60))"\n')
    os.environ['DEEPSWE_MODE'] = 'full'
    (report / 'prompt.txt').write_text(make_general_prompt(QWEN35_ID, 36000))
    with (report / 'prompt_environment.log').open('w') as log:
        run(container(image, probe_home, gpu=True, env=TRAIN_ENV) +
            ['bash', '/home/ben/setup_training_env.sh'], timeout=180, stdout=log)
        with cache_overlay(Path(os.environ['HF_HOME']), tmp / 'prompt_cache') as cache:
            run(container(image, probe_home, [f'{cache}:/home/ben/hf_cache',
                f'{ROOT / "dev_utils/deepswe/prompt_environment_probe.py"}:/audit.py:ro',
                f'{ROOT / "dev_utils/deepswe/prompt_network_probe.py"}:/network_probe.py:ro'],
                network=True, gpu=True, pwd='/home/ben/task', env=TRAIN_ENV) +
                ['bash', '-c', 'python /audit.py && python /network_probe.py'], timeout=600, stdout=log)
    (report / 'prompt_environment.json').write_bytes((task / 'prompt_environment.json').read_bytes())
    (report / 'prompt_network.json').write_bytes((task / 'prompt_network.json').read_bytes())
    # Run the exact prompt command through the real authenticated host bridge.
    bridge = TrialBridge(probe_home, report, {'train': image, 'eval': image}, suite)
    bridge.start()
    try:
        with (report / 'evaluate_client.log').open('w') as log:
            run(container(image, probe_home, network=True, pwd='/home/ben/task',
                env={**TRAIN_ENV, 'DEEPSWE_EVAL_URL': bridge.url,
                     'DEEPSWE_EVAL_TOKEN': bridge.agent_token}) +
                ['python', 'evaluate.py', '--model-path', 'final_model',
                 '--templates-dir', 'templates', '--limit', '1',
                 '--json-output-file', 'trial_metrics.json'], timeout=4500, stdout=log)
        metrics = json.loads((task / 'trial_metrics.json').read_text())
        assert set(metrics) == {'accuracy', 'num_tasks', 'num_tasks_completed',
                               'seed', 'max_tokens', 'evaluation_phase'}
    finally:
        bridge.close_training_window()
        bridge.server.shutdown()
        bridge.server.server_close()
    if metrics.get('accuracy') not in (0, 1):
        raise RuntimeError('End-to-end verifier produced no valid score')
    # Zero is a valid model result; the gate checks executable infrastructure,
    # not whether this deliberately tiny training update solves the task.
    (report / 'end_to_end.json').write_text(json.dumps(metrics, indent=2))
(report / 'ready.json').write_text(json.dumps(dict(model=QWEN35_ID, revision=QWEN35_REVISION,
    enable_thinking=True, prompt_environment=True, prompt_network=True, evaluation_client=True,
    validation_job_id=os.environ['SLURM_JOB_ID'], image=str(image), versions=json.loads((report / 'train_probe.json').read_text())['versions'], training_export=True, native_tool_call=True, end_to_end_evaluation=True, max_model_len=32768, output_budget=4000), indent=2))
print('Qwen3.5 real-model training/export and vLLM tool-call checks passed.')
