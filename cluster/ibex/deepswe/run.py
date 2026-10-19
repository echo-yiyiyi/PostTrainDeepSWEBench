"""Ibex post-training run (Codex -> checkpoint -> DeepSWE)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/eval/tasks/deepswe'))


import json
import os
import re
import shutil
import subprocess
import time

from runtime import ROOT, TASK, Bridge, container, run, EVAL_CONFIG, cache_overlay
from evaluation_config import PROTOCOL, TASK_ID, EXPERIMENT_MODE, BASE_COMMIT, HARNESS, MINI_REVISION
from training_support import make_prompt, save_artifacts, final_evaluation, prepare_teaching_materials


TRAIN_PATH = ('/home/ben/.venv/bin:/home/ben/.local/bin:/usr/local/nvidia/bin:'
              '/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/bin')
TRAIN_ENV = {'PATH': TRAIN_PATH, 'VIRTUAL_ENV': '/home/ben/.venv',
             'PYTHONNOUSERSITE': '1'}


def prepare_training_home(home, modern=False):
    uv = ROOT / '.cache/deepswe-tools/uv-0.9.7/uv'
    if not uv.is_file():
        raise RuntimeError('Run bash cluster/ibex/deepswe/prepare_uv.sh on the submission node first')
    (home / '.local/bin').mkdir(parents=True, exist_ok=True)
    shutil.copy2(uv, home / '.local/bin/uv')
    shutil.copy2(ROOT / 'cluster/ibex/deepswe/setup_training_env.sh', home / 'setup_training_env.sh')
    shutil.copytree(ROOT / 'containers/other_home_data/.codex', home / '.codex')
    codex_config = home / '.codex/config.toml'
    # Update the shell policy too: Codex overrides its inherited environment here.
    config = codex_config.read_text()
    if modern:
        # The Qwen3.5 image owns its CUDA stack. Do not replay old smoke-image
        # version labels or replace Singularity's injected library search path.
        config = re.sub(r'(?:NV_CUDA_LIB_VERSION|NV_LIBNCCL_PACKAGE|LD_LIBRARY_PATH) = "[^"]*",?\s*', '', config)

    config, count = re.subn(r'\bPATH = "[^"]*"', 'PATH = ' + json.dumps(TRAIN_PATH), config)
    if count != 1:
        raise RuntimeError('Expected exactly one Codex shell PATH setting')
    additions = ', '.join(f'{key} = {json.dumps(value)}' for key, value in TRAIN_ENV.items()
                          if key not in ('PATH', 'PYTHONNOUSERSITE'))
    config = config.replace('set = {', 'set = {' + additions + ', ', 1)
    codex_config.write_text('model_reasoning_effort = "low"\n\n' + config)


def stop_monitor(process):
    import signal
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()


def main():
    started = time.monotonic()
    full = os.environ.get('DEEPSWE_MODE') == 'full'
    train_seconds = int(os.environ.get('DEEPSWE_TRAIN_SECONDS', '36000' if full else '3600'))
    if not full:
        train_seconds = min(train_seconds, 3600)
    if train_seconds <= 0:
        raise ValueError("DEEPSWE_TRAIN_SECONDS must be positive")
    result = Path(os.environ['DEEPSWE_RESULTS']).resolve() / os.environ['SLURM_JOB_ID']
    result.mkdir(parents=True, exist_ok=False)
    home = result / 'training_home'
    task = home / 'task'
    task.mkdir(parents=True)
    images = {name: Path(os.environ[f'DEEPSWE_{name.upper()}_SIF']).resolve()
              for name in (('train', 'eval') if full else ('train', 'eval', 'task'))}
    for image in images.values():
        if not image.is_file():
            raise FileNotFoundError(image)
    base = os.environ.get('DEEPSWE_BASE_MODEL', 'Qwen/Qwen3.5-9B' if full else 'Qwen/Qwen3-1.7B-Base')
    config = {'agent': 'codex', 'agent_config': 'gpt-6-sol', 'model_to_train': base,
              'reasoning_effort': 'low', 'train_seconds': train_seconds,
              'slurm_walltime': os.environ.get('DEEPSWE_JOB_WALLTIME', '12:00:00' if full else '01:30:00'),
              'container_fakeroot': True,
              'benchmark': 'deepswe_smoke', 'num_tasks': 1, 'seed': 72332,
              'task_id': TASK_ID, 'experiment_mode': EXPERIMENT_MODE,
              'harness': HARNESS, 'mini_revision': MINI_REVISION,
              'training_task_exposed': True, 'reference_solution_exposed': True,
              'verifier_tests_exposed': True, 'held_out_evaluation': False,
              'single_task_test': True, 'integrity_judges': 'shared run_all_judges',
              'images': {k: str(v) for k, v in images.items()}}
    if full:
        from full_suite import load_manifest
        from model_profiles import QWEN35_ID, QWEN35_REVISION
        if base != QWEN35_ID:
            raise ValueError('The full-suite profile currently targets Qwen/Qwen3.5-9B')
        suite = load_manifest(Path(os.environ['DEEPSWE_SUITE_MANIFEST']))
        shutil.copy2(os.environ['DEEPSWE_SUITE_MANIFEST'], result / 'suite_manifest.json')
        config.update(benchmark='deepswe', num_tasks=len(suite['tasks']),
                      task_id=None, experiment_mode='full_held_out', single_task_test=False,
                      training_task_exposed=False, reference_solution_exposed=False,
                      verifier_tests_exposed=False, held_out_evaluation=True,
                      evaluation_timing='training_trials_and_final', model_revision=QWEN35_REVISION,
                      api_usage_policy='limited_pipeline_assistance')
    (result / 'run_config.json').write_text(json.dumps(config, indent=2))
    # Verify task image and no-network support before spending time training.
    if not full:
        run(container(images['task'], result / 'preflight_home') +
        ['bash', '-c', f'test -d /app/.git && test "$(git -C /app -c safe.directory=/app rev-parse HEAD)" = {BASE_COMMIT} && node --version && python3 --version'])
    for source, destination in [
        ('agents/codex/solve.sh', home / 'agent_solve.sh'),
        ('cluster/ibex/deepswe/start_agent.sh', home / 'start_agent.sh'),
        ('src/utils/update_agent_cli.sh', home / 'update_agent_cli.sh'),
        ('src/eval/tasks/' + ('deepswe' if full else 'deepswe_smoke') + '/evaluate.py', task / 'evaluate.py'),
        ('src/judges/judge_tools/contamination_check.py', home / 'contamination_check.py')]:
        shutil.copy2(ROOT / source, destination)
    # Derive the checker reference from the vendored prompt (test_data.json is
    # globally gitignored, so a fresh checkout must not depend on a local copy).
    if not full:
        (home / 'test_data.json').write_text(json.dumps([{
            'id': TASK_ID,
            'question': (TASK / 'instruction.md').read_text()}], indent=2))
    prepare_training_home(home, modern=full)
    with (result / 'training_env.log').open('w') as log:
        run(container(images['train'], home, gpu=True, env={
            **TRAIN_ENV, 'DEEPSWE_REQUIRE_CUDA': '1'}) +
            ['bash', '/home/ben/setup_training_env.sh'], timeout=120, stdout=log)
    if full:
        from trial_bridge import TrialBridge
        bridge = TrialBridge(home, result, images, suite)
    else:
        bridge = Bridge(home, result, images)
    if bridge:
        bridge.start()
    shutil.copytree(ROOT / 'src/eval/templates', task / 'templates')
    protocol = PROTOCOL
    if full:
        protocol = protocol.replace('templates/qwen3.jinja', 'templates/qwen3_5.jinja')
        protocol += '\nQwen3.5 uses XML function/parameter blocks inside tool_call, not Qwen3 JSON tool blocks. Always render conversations with the provided template and enable_thinking=False. Training trials use evaluate.py and return aggregate scores; final evaluation covers all 113 tasks after training.\n'
    (task / 'evaluation_protocol.txt').write_text(protocol)
    shutil.copy2(ROOT / 'src/eval/tasks/deepswe/mini_swe/mini.yaml', task / 'mini_swe_config.yaml')
    shutil.copy2(ROOT / 'src/eval/tasks/deepswe/mini_swe/bash_tool.json', task / 'mini_swe_bash_tool.json')
    public_eval_config = dict(EVAL_CONFIG)
    if full:
        public_eval_config.update(gpu_memory_utilization=0.8, enable_thinking=False,
                                  model_to_train=base, num_tasks=113, training_trials=True)
    (task / 'evaluation_config.json').write_text(json.dumps(public_eval_config, indent=2))
    if full:
        (task / 'model_profile.json').write_text(json.dumps(dict(model=base, revision=QWEN35_REVISION,
            chat_template='templates/qwen3_5.jinja', enable_thinking=False,
            cached_model='/home/ben/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/' + QWEN35_REVISION), indent=2))
    else:
        prepare_teaching_materials(task)
    prompt = make_prompt(base, train_seconds, mode='general' if full else EXPERIMENT_MODE)
    (result / 'prompt.txt').write_text(prompt)
    (home / 'training_prompt.txt').write_text(prompt)
    cache = Path(os.environ['HF_HOME']).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    env = {'AGENT_CONFIG': 'gpt-6-sol', 'CODEX_API_KEY': os.environ['OPENAI_API_KEY'],
           'HF_HOME': '/home/ben/hf_cache', 'NUM_GPUS': '1', 'PYTHONNOUSERSITE': '1',
           **TRAIN_ENV}
    if bridge:
        env.update(DEEPSWE_EVAL_URL=bridge.url, DEEPSWE_EVAL_TOKEN=bridge.agent_token)
    if full:
        env['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
    for key in ('CODEX_CLI_VERSION', 'POST_TRAIN_BENCH_SKIP_CLI_UPDATE'):
        if key in os.environ:
            env[key] = os.environ[key]
    status = {'status': 'running', 'single_task_test': not full}
    monitor = subprocess.Popen(['bash', str(ROOT / 'src/utils/system_monitor.sh')], cwd=result,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        # Same remaining-time interface as PostTrainBench, with a minute-scale budget.
        deadline = int(time.time()) + train_seconds
        (task / 'timer.sh').write_text(
            '#!/bin/bash\n'
            f'remaining=$(({deadline} - $(date +%s)))\n'
            'if (( remaining <= 0 )); then echo "Timer expired!"; else\n'
            'printf "Remaining time (minutes:seconds): %d:%02d\\n" '
            '"$((remaining / 60))" "$((remaining % 60))"; fi\n')
        training_started = time.monotonic()
        with (result / 'solve_out.txt').open('w') as log:
            # Record timeout as 124; run() terminates the whole container process group.
            try:
                with cache_overlay(cache, result / 'training_cache_overlay') as merged_cache:
                    cmd = container(images['train'], home, [f'{merged_cache}:/home/ben/hf_cache'],
                                    network=True, gpu=True, pwd='/home/ben/task', env=env)
                    remaining = train_seconds - (time.monotonic() - training_started)
                    code, _ = run(cmd + ['bash', '/home/ben/start_agent.sh'],
                                  timeout=max(0, remaining), check=False, stdout=log)
            except subprocess.TimeoutExpired:
                code = 124
        status['agent_exit_code'] = code
        if bridge:
            bridge.close_training_window()
        save_artifacts(result, home, time.monotonic() - training_started)
        stop_monitor(monitor)
        status['phase'] = 'judges'
        (result / 'status.json').write_text(json.dumps(status, indent=2))
        with (result / 'judges.log').open('w') as log:
            subprocess.run(['bash', str(ROOT / 'src/judges/deepswe/judge.sh'), str(result)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['judges_complete'] = json.loads((result / 'judges_status.json').read_text())['complete']
        if not (task / 'final_model/config.json').is_file():
            raise RuntimeError('Codex did not produce final_model/config.json')
        status['phase'] = 'awaiting_final_evaluation' if full else 'final_evaluation'
        (result / 'status.json').write_text(json.dumps(status, indent=2))
        if not full:
            final_evaluation(result, home)
        (result / 'final_model').symlink_to(task / 'final_model', target_is_directory=True)
        status['status'] = 'training_completed' if full else 'completed'
    except Exception as error:
        status.update(status='infrastructure_or_training_failure', error=str(error))
        raise
    finally:
        stop_monitor(monitor)
        status['elapsed_seconds'] = round(time.monotonic() - started, 2)
        (result / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        if bridge:
            bridge.server.shutdown()
            bridge.server.server_close()


if __name__ == '__main__':
    main()
