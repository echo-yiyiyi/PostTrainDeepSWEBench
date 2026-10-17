"""Ibex 30-minute post-training smoke run (Codex -> checkpoint -> DeepSWE)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from runtime import ROOT, TASK, Bridge, container, run


TRAIN_PATH = ('/home/ben/.venv/bin:/home/ben/.local/bin:/usr/local/nvidia/bin:'
              '/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/bin')
TRAIN_ENV = {'PATH': TRAIN_PATH, 'VIRTUAL_ENV': '/home/ben/.venv',
             'UV_PYTHON_DOWNLOADS': 'never',
             'UV_CONSTRAINT': '/home/ben/training-constraints.txt',
             'PYTHONNOUSERSITE': '1'}


def prepare_training_home(home):
    uv = ROOT / '.cache/deepswe-tools/uv-0.9.7/uv'
    if not uv.is_file():
        raise RuntimeError('Run bash src/deepswe/prepare_uv.sh on the submission node first')
    (home / '.local/bin').mkdir(parents=True, exist_ok=True)
    shutil.copy2(uv, home / '.local/bin/uv')
    shutil.copy2(ROOT / 'src/deepswe/setup_training_env.sh', home / 'setup_training_env.sh')
    shutil.copytree(ROOT / 'containers/other_home_data/.codex', home / '.codex')
    codex_config = home / '.codex/config.toml'
    # Update the shell policy too: Codex overrides its inherited environment here.
    config = codex_config.read_text()
    config, count = re.subn(r'\bPATH = "[^"]*"', 'PATH = ' + json.dumps(TRAIN_PATH), config)
    if count != 1:
        raise RuntimeError('Expected exactly one Codex shell PATH setting')
    additions = ', '.join(f'{key} = {json.dumps(value)}' for key, value in TRAIN_ENV.items()
                          if key not in ('PATH', 'PYTHONNOUSERSITE'))
    config = config.replace('set = {', 'set = {' + additions + ', ', 1)
    codex_config.write_text('model_reasoning_effort = "low"\n\n' + config)


def main():
    started = time.monotonic()
    result = Path(os.environ['DEEPSWE_RESULTS']).resolve() / os.environ['SLURM_JOB_ID']
    result.mkdir(parents=True, exist_ok=False)
    home = result / 'training_home'
    task = home / 'task'
    task.mkdir(parents=True)
    images = {name: Path(os.environ[f'DEEPSWE_{name.upper()}_SIF']).resolve()
              for name in ('train', 'eval', 'task')}
    for image in images.values():
        if not image.is_file():
            raise FileNotFoundError(image)
    base = os.environ.get('DEEPSWE_BASE_MODEL', 'Qwen/Qwen3-1.7B-Base')
    config = {'agent': 'codex', 'agent_config': 'gpt-5-mini', 'model_to_train': base,
              'reasoning_effort': 'low',
              'benchmark': 'deepswe', 'num_tasks': 1, 'seed': 72332,
              'smoke_only': True, 'integrity_judges': 'not run in 30-minute smoke test',
              'images': {k: str(v) for k, v in images.items()}}
    (result / 'run_config.json').write_text(json.dumps(config, indent=2))
    # Verify task image and no-network support before spending time training.
    run(container(images['task'], result / 'preflight_home') +
        ['bash', '-c', 'test -d /app/.git && python -c "import pytest, fastapi"'])
    for source, destination in [
        ('agents/codex/solve.sh', home / 'agent_solve.sh'),
        ('src/deepswe/start_agent.sh', home / 'start_agent.sh'),
        ('src/utils/update_agent_cli.sh', home / 'update_agent_cli.sh'),
        ('src/eval/tasks/deepswe/evaluate.py', task / 'evaluate.py'),
        ('src/judges/judge_tools/contamination_check.py', home / 'contamination_check.py')]:
        shutil.copy2(ROOT / source, destination)
    # Derive the checker reference from the vendored prompt (test_data.json is
    # globally gitignored, so a fresh checkout must not depend on a local copy).
    (home / 'test_data.json').write_text(json.dumps([{
        'id': 'fastapi-implicit-head-options',
        'question': (TASK / 'instruction.md').read_text()}], indent=2))
    prepare_training_home(home)
    with (result / 'training_env.log').open('w') as log:
        run(container(images['train'], home, gpu=True, env={
            **TRAIN_ENV, 'DEEPSWE_REQUIRE_CUDA': '1'}) +
            ['bash', '/home/ben/setup_training_env.sh'], timeout=120, stdout=log)
    bridge = Bridge(home, result, images)
    bridge.start()
    # Reserve at least 12 minutes for loading, rollout and verification, plus cleanup.
    train_seconds = min(int(os.environ.get('DEEPSWE_TRAIN_SECONDS', '720')), 840)
    prompt = f'''You are the PostTrainBench training agent. Improve {base} on general
software-engineering repository repair tasks, evaluated on a held-out DeepSWE task.
You are NOT the model solving the evaluation task. Train only the specified base
model; save a complete Hugging Face model AND tokenizer in /home/ben/task/final_model.
The evaluation uses Qwen3 chat formatting and a JSON shell-action scaffold:
{{"command":"bash command"}}, or {{"done":true}}. The evaluated model edits /app
and must git commit its changes before finishing. There are at most 12 turns.
You have {train_seconds // 60} minutes, on the GPU assigned by Slurm (inspect nvidia-smi).
Write a valid checkpoint early, then improve it. If using LoRA, merge adapters into
full model weights. Do not submit an adapter-only checkpoint. Save tokenizer files.
Python uses /home/ben/.venv, a writable virtual environment that inherits the
image's preinstalled training packages. You have no root access. Install extras
with uv pip install --python /home/ben/.venv/bin/python PACKAGE; never use --system
or --user. UV_CONSTRAINT pins the existing training stack; do not upgrade it.
final_model must load in the original evaluation image.
Use independently sourced/synthesized training data;
never train on DeepSWE questions, repositories' task-specific solutions, hidden
verifier tests, or derivatives. Do not look up PostTrainBench solutions/traces.
Do not call external LLM APIs with credentials: they are for running you only.
The test_data.json in your home is ONLY for checking contamination with
python ../contamination_check.py --reference ../test_data.json --input train_data.jsonl.
Work in /home/ben/task. Do not read unrelated paths. Do not attempt to start containers.
Optional evaluation: free GPU memory, then python evaluate.py --model-path final_model
--json-output-file trial_metrics.json. This calls a host service; evaluation can
consume most of this short test's budget, so prioritize producing a checkpoint.
The outer harness will evaluate your final checkpoint once after you exit.
Write training provenance and steps in training_report.json. Finish before your deadline.
'''
    (result / 'prompt.txt').write_text(prompt)
    (home / 'training_prompt.txt').write_text(prompt)
    cache = Path(os.environ['HF_HOME']).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    env = {'AGENT_CONFIG': 'gpt-5-mini', 'CODEX_API_KEY': os.environ['OPENAI_API_KEY'],
           'HF_HOME': '/home/ben/hf_cache', 'NUM_GPUS': '1', 'PYTHONNOUSERSITE': '1',
           'DEEPSWE_EVAL_URL': bridge.url, 'DEEPSWE_EVAL_TOKEN': bridge.agent_token,
           **TRAIN_ENV}
    for key in ('CODEX_CLI_VERSION', 'POST_TRAIN_BENCH_SKIP_CLI_UPDATE'):
        if key in os.environ:
            env[key] = os.environ[key]
    status = {'status': 'running', 'smoke_only': True}
    try:
        cmd = container(images['train'], home, [f'{cache}:/home/ben/hf_cache'],
                        network=True, gpu=True, pwd='/home/ben/task', env=env)
        with (result / 'solve_out.txt').open('w') as log:
            # GNU timeout returns 124; terminate the complete process group on timeout.
            try:
                code, _ = run(cmd + ['bash', '/home/ben/start_agent.sh'],
                              timeout=train_seconds, check=False, stdout=log)
            except subprocess.TimeoutExpired:
                code = 124
        status['agent_exit_code'] = code
        bridge.accept_agent = False
        # Any in-flight exploratory evaluation must finish before the final one.
        if not bridge.lock.acquire(timeout=30):
            raise RuntimeError('Exploratory evaluation still running; final evaluation not started')
        bridge.lock.release()
        if not (task / 'final_model/config.json').is_file():
            raise RuntimeError('Codex did not produce final_model/config.json')
        if time.monotonic() - started > 1020:
            raise RuntimeError('Insufficient time remaining for final evaluation')
        metrics = bridge.evaluate('/home/ben/task/final_model')
        (result / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        (result / 'final_model').symlink_to(task / 'final_model', target_is_directory=True)
        status['status'] = 'completed'
    except Exception as error:
        status.update(status='infrastructure_or_training_failure', error=str(error))
        raise
    finally:
        status['elapsed_seconds'] = round(time.monotonic() - started, 2)
        (result / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        bridge.server.shutdown()
        bridge.server.server_close()


if __name__ == '__main__':
    main()
