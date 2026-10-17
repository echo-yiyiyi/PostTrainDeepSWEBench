"""Host-side Singularity task execution and trusted evaluation bridge.

Only the pinned FastAPI task is supported. This is a native adapter for its
Harbor task contract, not a general Pier environment backend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'src/eval/tasks/deepswe/harbor_task'
BASE = '11614be9021aa4ac078d4d0693a8b5250a1010d8'


def clean_env():
    # Singularity inherits explicit SINGULARITYENV_* even with --cleanenv.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SINGULARITY', 'APPTAINER'))}
    # Only forward the temporary directory created by our Slurm entry point.
    if os.environ.get('DEEPSWE_CONTAINER_TMPDIR'):
        env['SINGULARITY_TMPDIR'] = os.environ['DEEPSWE_CONTAINER_TMPDIR']
    return env


def run(cmd, timeout=120, check=True, stdout=None):
    """Kill the entire container process group on timeout, not only its launcher."""
    proc = subprocess.Popen(cmd, env=clean_env(), start_new_session=True,
                            stdout=stdout or subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except BaseException:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise
    if check and proc.returncode:
        raise RuntimeError(f'Command failed ({proc.returncode}): {output or "see run log"}')
    return proc.returncode, output or ''


def container(image, home, binds=(), network=False, gpu=False, pwd='/tmp', env=None,
              fakeroot=False):
    home.mkdir(parents=True, exist_ok=True)
    cmd = ['singularity', 'exec', '--containall', '--cleanenv', '--no-mount', 'hostfs',
           '--home', f'{home}:/home/ben', '--pwd', pwd, '--writable-tmpfs']
    if fakeroot:
        cmd += ['--fakeroot']
    if not network:
        cmd += ['--net', '--network', 'none']
    if gpu:
        cmd += ['--nv']
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            cmd += ['--env', 'CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES']]
    for bind in binds:
        cmd += ['--bind', bind]
    for key, value in (env or {}).items():
        cmd += ['--env', f'{key}={value}']
    return cmd + [str(image)]


def resolve_checkpoint(model, home):
    """Never let an agent-controlled request mount arbitrary host paths."""
    path = Path(model)
    relative = path.relative_to('/home/ben/task')
    checkpoint = (home / 'task' / relative).resolve()
    if not checkpoint.is_relative_to((home / 'task').resolve()):
        raise ValueError('Checkpoint escapes task directory')
    if not (checkpoint / 'config.json').is_file():
        raise ValueError('Expected a complete Hugging Face checkpoint with config.json')
    # A checkpoint may not smuggle symlinks to files outside the writable task.
    for entry in checkpoint.rglob('*'):
        if entry.is_symlink() and not entry.resolve().is_relative_to((home / 'task').resolve()):
            raise ValueError('Checkpoint contains an external symlink')
    return checkpoint


class TaskEnvironment:
    def __init__(self, directory, image):
        self.directory, self.image = directory, image
        self.app = directory / 'agent_app'
        self.home = directory / 'agent_home'
        self.app.mkdir(parents=True)
        self.extract(self.app, self.home)

    def extract(self, target, home):
        target.mkdir(parents=True, exist_ok=True)
        run(container(self.image, home, [f'{target}:/export'], fakeroot=True) +
            ['bash', '-c', 'cp -a /app/. /export/'], timeout=180)

    def shell(self, command):
        cmd = container(self.image, self.home, [f'{self.app}:/app'], pwd='/app', fakeroot=True)
        try:
            code, output = run(cmd + ['bash', '-c', command], timeout=60, check=False)
        except subprocess.TimeoutExpired:
            return {'exit_code': 124, 'output': 'Command exceeded the 60-second limit and was terminated.'}
        return {'exit_code': code, 'output': output[-16000:]}

    def verify(self):
        # Collect exactly the committed diff, as specified by [[verifier.collect]].
        artifacts = self.directory / 'artifacts'
        artifacts.mkdir()
        with (artifacts / 'model.patch').open('w') as patch:
            run(container(self.image, self.home, [f'{self.app}:/app'], pwd='/app', fakeroot=True) +
                ['git', '-c', 'safe.directory=/app', 'diff', '--binary', BASE, 'HEAD'], stdout=patch)
        app = self.directory / 'verifier_app'
        home = self.directory / 'verifier_home'
        logs = self.directory / 'verifier'
        logs.mkdir()
        self.extract(app, home)
        cmd = container(self.image, home, [f'{app}:/app', f'{TASK / "tests"}:/tests:ro',
            f'{artifacts}:/logs/artifacts:ro', f'{logs}:/logs/verifier'], pwd='/app', fakeroot=True)
        with (logs / 'test-stdout.txt').open('w') as out:
            run(cmd + ['bash', '/tests/test.sh'], timeout=240, stdout=out)
        reward = json.loads((logs / 'reward.json').read_text())
        if reward.get('reward') not in (0, 1):
            raise RuntimeError('Verifier did not produce a valid binary reward')
        return reward


class Bridge:
    def __init__(self, home, result, images):
        self.home, self.result, self.images = home, result, images
        self.agent_token = secrets.token_urlsafe(32)
        self.eval_token = None
        self.task_env = None
        self.lock = threading.Lock()
        self.accept_agent = True
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), self.handler())
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    token = self.headers.get('Authorization', '').removeprefix('Bearer ')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 65536:
                        raise ValueError('Invalid request length')
                    body = json.loads(self.rfile.read(size))
                    if self.path == '/evaluate' and secrets.compare_digest(token, bridge.agent_token):
                        if not bridge.accept_agent:
                            raise ValueError('Training evaluation window has closed')
                        value = bridge.evaluate(body['model_path'], int(body.get('seed', 72332)))
                    elif bridge.eval_token and secrets.compare_digest(token, bridge.eval_token):
                        if self.path == '/shell':
                            value = bridge.task_env.shell(body['command'])
                        elif self.path == '/verify':
                            value = bridge.task_env.verify()
                        else:
                            raise ValueError('Unknown evaluation operation')
                    else:
                        self.send_error(403)
                        return
                    data = json.dumps(value).encode()
                    self.send_response(200)
                except Exception as error:
                    data = json.dumps({'error': str(error)}).encode()
                    self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except BrokenPipeError:
                    pass
        return Handler

    def evaluate(self, model_path, seed=72332):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('An evaluation is already running')
        try:
            checkpoint = resolve_checkpoint(model_path, self.home)
            directory = self.result / ('eval_' + secrets.token_hex(6))
            directory.mkdir()
            self.task_env = TaskEnvironment(directory, self.images['task'])
            self.eval_token = secrets.token_urlsafe(32)
            cache = Path(os.environ['HF_HOME']).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            cmd = container(self.images['eval'], directory / 'model_home', [
                f'{checkpoint}:/model:ro', f'{ROOT / "src/deepswe"}:/harness:ro',
                f'{ROOT / "src/eval/per_sample_seed.py"}:/harness_seed.py:ro',
                f'{ROOT / "src/eval/tasks/deepswe/evaluate_final_eval.py"}:/evaluate_final_eval.py:ro',
                f'{ROOT / "src/eval/templates"}:/templates:ro',
                f'{TASK / "instruction.md"}:/instruction.md:ro',
                f'{directory}:/output', f'{cache}:/hf_cache'], network=True, gpu=True,
                env={'DEEPSWE_WORKER_URL': self.url, 'DEEPSWE_WORKER_TOKEN': self.eval_token,
                     'HF_HOME': '/hf_cache', 'VLLM_API_KEY': 'inspectai', 'PYTHONNOUSERSITE': '1'})
            with (directory / 'evaluate.log').open('w') as log:
                run(cmd + ['python', '/evaluate_final_eval.py', '--seed', str(seed)], timeout=720, stdout=log)
            metrics = json.loads((directory / 'metrics.json').read_text())
            metrics['evaluation_directory'] = directory.name
            return metrics
        finally:
            self.eval_token = None
            self.task_env = None
            self.lock.release()

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
