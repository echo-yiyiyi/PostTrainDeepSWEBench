"""Native mini + fixed vLLM image + task/verifier smoke, no teacher API calls.

An older checkpoint is only a transport probe, NOT a comparable benchmark score.
"""
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/deepswe'))
from runtime import Bridge, EVAL_CONFIG, TaskEnvironment, cache_overlay
from evaluation_config import TASK_ID, EXPERIMENT_MODE

directory = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('mini_smoke_' + os.environ['SLURM_JOB_ID'])
directory.mkdir()
checkpoint = Path(sys.argv[1]).resolve()
assert (checkpoint / 'config.json').is_file()
images = {name: Path(os.environ[f'DEEPSWE_{name.upper()}_SIF']) for name in ('eval', 'task')}
bridge = Bridge(directory / 'unused_training_home', directory, images)
bridge.accept_agent = False
bridge.eval_token = secrets.token_urlsafe(32)
config = dict(EVAL_CONFIG, max_tokens=128, agent_timeout_seconds=180,
              task_id=TASK_ID, experiment_mode='infrastructure_smoke', infrastructure_smoke=True)
(directory / 'evaluation_config.json').write_text(json.dumps(config, indent=2))
bridge.task_env = TaskEnvironment(directory, images['task'], bridge.cancel)
bridge.start()
try:
    with cache_overlay(Path(os.environ['HF_HOME']), directory / 'cache_overlay') as cache:
        metrics = bridge._run_mini_model(checkpoint, directory, cache, 72332, config)
    assert metrics['harness'] == 'mini-swe-agent'
    assert metrics['accuracy'] in (0, 1)
    assert (directory / 'mini-swe-agent.trajectory.json').is_file()
    print('PASS: native mini agent, vLLM API, Singularity task and separate official verifier', flush=True)
    print(json.dumps(metrics, indent=2))
finally:
    bridge.server.shutdown()
    bridge.server.server_close()
