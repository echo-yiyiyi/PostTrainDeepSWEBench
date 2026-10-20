"""Host-side Singularity task execution and trusted evaluation bridge.

Supports the smoke task and manifest-selected full-suite tasks through their
Harbor task contracts. This is not a general Pier environment backend.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
import shutil
import tempfile
import os
from pathlib import Path
import secrets
import shlex
import socket
import signal
import subprocess
import threading
import time
import urllib.request

from evaluation_config import defaults, TASK_DIRECTORY, BASE_COMMIT, TASK_ID, EXPERIMENT_MODE
from model_profiles import serving_args
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[4]
TASK = ROOT / 'src/eval/tasks/deepswe_smoke' / TASK_DIRECTORY
BASE = BASE_COMMIT
EVAL_CONFIG = defaults(TASK / 'task.toml')


def clean_env():
    # Singularity inherits explicit SINGULARITYENV_* even with --cleanenv.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('SINGULARITY', 'APPTAINER'))}
    # Only forward the temporary directory created by our Slurm entry point.
    if os.environ.get('DEEPSWE_CONTAINER_TMPDIR'):
        env['SINGULARITY_TMPDIR'] = os.environ['DEEPSWE_CONTAINER_TMPDIR']
    return env


def run(cmd, timeout=120, check=True, stdout=None, stderr=subprocess.STDOUT, cancel=None, env=None):
    """Kill the entire container process group on timeout, not only its launcher."""
    proc = subprocess.Popen(cmd, env=clean_env() if env is None else env, start_new_session=True,
                            stdout=stdout or subprocess.PIPE, stderr=stderr, text=True)
    try:
        deadline = time.monotonic() + timeout
        while True:
            if cancel is not None and cancel.is_set():
                raise RuntimeError('Evaluation cancelled when training ended')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout)
            try:
                output, _ = proc.communicate(timeout=min(remaining, 0.5) if cancel else remaining)
                break
            except subprocess.TimeoutExpired:
                if cancel is None or time.monotonic() >= deadline:
                    raise
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        raise
    if check and proc.returncode:
        raise RuntimeError(f'Command failed ({proc.returncode}): {output or "see run log"}')
    return proc.returncode, output or ''


@contextmanager
def cache_overlay(lower, directory):
    """Same read-only shared cache plus private writes as PostTrain's overlay."""
    directory.mkdir(parents=True, exist_ok=False)
    upper, work = [directory / name for name in ('upper', 'work')]
    for path in (upper, work):
        path.mkdir()
    # Ibex forbids FUSE mountpoints on its shared filesystem. Keep the mount
    # on node-local /tmp, while cache writes stay in the configured result storage.
    with tempfile.TemporaryDirectory(prefix='deepswe-hf-',
            dir=os.environ.get('DEEPSWE_CONTAINER_TMPDIR', '/tmp')) as local:
        merged = Path(local) / 'merged'
        merged.mkdir()
        run(['fuse-overlayfs', '-o', f'lowerdir={lower},upperdir={upper},workdir={work}', str(merged)])
        try:
            yield merged
        finally:
            run(['fusermount', '-u', str(merged)])
    shutil.rmtree(directory)


def container(image, home, binds=(), network=False, gpu=False, pwd='/tmp', env=None):
    home.mkdir(parents=True, exist_ok=True)
    env = dict(env or {})
    if gpu:
        # Image /root and /tmp may be backed by Singularity's small session
        # filesystem. Compilation artifacts need the disk-backed private home.
        (home / 'tmp').mkdir(exist_ok=True)
        for name in ('triton', 'torchinductor', 'vllm', 'cuda', 'uv'):
            (home / '.cache' / name).mkdir(parents=True, exist_ok=True)
        for key, value in {
            'TMPDIR': '/home/ben/tmp',
            'XDG_CACHE_HOME': '/home/ben/.cache',
            # The vLLM image sets UV_CACHE_DIR=/opt/uv/cache (read-only in SIF).
            'UV_CACHE_DIR': '/home/ben/.cache/uv',
            'TRITON_CACHE_DIR': '/home/ben/.cache/triton',
            'TORCHINDUCTOR_CACHE_DIR': '/home/ben/.cache/torchinductor',
            'VLLM_CACHE_ROOT': '/home/ben/.cache/vllm',
            'CUDA_CACHE_PATH': '/home/ben/.cache/cuda',
        }.items():
            env.setdefault(key, value)
    cmd = ['singularity', 'exec', '--fakeroot', '--containall', '--cleanenv', '--no-mount', 'hostfs',
           '--home', f'{home}:/home/ben', '--pwd', pwd, '--writable-tmpfs']
    if not network:
        cmd += ['--net', '--network', 'none']
    if gpu:
        cmd += ['--nv']
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            # Singularity 3.9 parses --env as CSV: 0,1 otherwise becomes 0.
            # A per-invocation file also avoids races between evaluator launches.
            with tempfile.NamedTemporaryFile(mode='w', prefix='.cuda-env-',
                                             dir=home, delete=False) as env_file:
                env_file.write('CUDA_VISIBLE_DEVICES=' +
                               shlex.quote(os.environ['CUDA_VISIBLE_DEVICES']) + '\n')
            cmd += ['--env-file', env_file.name]
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
    def __init__(self, directory, image, cancel=None, task=TASK, base=BASE):
        self.directory, self.image = directory, image
        self.task, self.base = Path(task), base
        self.config = defaults(self.task / 'task.toml')
        self.cancel = cancel
        self.app = directory / 'agent_app'
        self.home = directory / 'agent_home'
        self.app.mkdir(parents=True)
        self.extract(self.app, self.home)
        # Identity is environment setup, not a task the evaluated model must solve.
        # Persist only in this attempt's repository; leave commit creation to the
        # agent and keep the verifier's committed-diff collection unchanged.
        with (directory / 'git_setup.log').open('w') as log:
            run(container(self.image, self.home, [f'{self.app}:/app'], pwd='/app') +
                ['bash', '-e', '-c',
                 'git -C /app -c safe.directory=/app config --local user.name "DeepSWE Agent"\n'
                 'git -C /app -c safe.directory=/app config --local user.email "agent@deepswe.local"'],
                stdout=log, timeout=60, cancel=self.cancel)

    def extract(self, target, home):
        target.mkdir(parents=True, exist_ok=True)
        run(container(self.image, home, [f'{target}:/export']) +
            ['bash', '-c', 'cp -a /app/. /export/'], timeout=180, cancel=self.cancel)

    def shell(self, command, timeout=60, env=None, cwd='/app', truncate=True):
        cmd = container(self.image, self.home, [f'{self.app}:/app'], pwd=cwd, env=env)
        try:
            # Merge the COMMAND's stderr inside the container, before running it.
            # The launcher's stderr stays separate: mini-swe checks the first
            # output line for its submission marker. Pass the command as data,
            # not interpolated shell text, so quoting/redirections stay intact.
            with (self.directory / 'singularity-shell.log').open('a') as diagnostic:
                code, output = run(cmd + ['bash', '-c',
                    'exec 2>&1\nexec bash -c "$1"', 'deepswe-shell', command],
                    timeout=timeout, check=False, cancel=self.cancel, stderr=diagnostic)
        except subprocess.TimeoutExpired:
            return {'exit_code': 124, 'output': '', 'exception_info': f'Command exceeded {timeout} seconds and was terminated.'}
        return {'exit_code': code, 'output': output[-16000:] if truncate else output}

    def verify(self):
        # Collect exactly the committed diff, as specified by [[verifier.collect]].
        artifacts = self.directory / 'artifacts'
        artifacts.mkdir()
        with (artifacts / 'model.patch').open('w') as patch, (artifacts / 'collect.log').open('w') as diagnostic:
            run(container(self.image, self.home, [f'{self.app}:/app'], pwd='/app') +
                ['git', '-c', 'safe.directory=/app', 'diff', '--binary', self.base, 'HEAD'], stdout=patch, stderr=diagnostic, timeout=300, cancel=self.cancel)
        app = self.directory / 'verifier_app'
        home = self.directory / 'verifier_home'
        logs = self.directory / 'verifier'
        logs.mkdir()
        self.extract(app, home)
        cmd = container(self.image, home, [f'{app}:/app', f'{self.task / "tests"}:/tests:ro',
            f'{artifacts}:/logs/artifacts:ro', f'{logs}:/logs/verifier'], pwd='/app')
        with (logs / 'test-stdout.txt').open('w') as out:
            run(cmd + ['bash', '/tests/test.sh'], timeout=self.config['verifier_timeout_seconds'], stdout=out, cancel=self.cancel)
        reward = json.loads((logs / 'reward.json').read_text())
        if reward.get('reward') not in (0, 1):
            raise RuntimeError('Verifier did not produce a valid binary reward')
        return reward


class Bridge:
    def __init__(self, home, result, images, suite_task=None):
        self.home, self.result, self.images = home, result, images
        self.suite_task = suite_task
        self.agent_token = secrets.token_urlsafe(32)
        self.eval_token = None
        self.task_env = None
        self.lock = threading.Lock()
        self.accept_agent = True
        self.cancel = threading.Event()
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
                    if not 0 < size <= 4 * 1024 * 1024:
                        raise ValueError('Invalid request length')
                    body = json.loads(self.rfile.read(size))
                    if self.path == '/evaluate' and secrets.compare_digest(token, bridge.agent_token):
                        if not bridge.accept_agent:
                            raise ValueError('Training evaluation window has closed')
                        value = bridge.evaluate_request(body)
                    elif bridge.eval_token and secrets.compare_digest(token, bridge.eval_token):
                        if self.path == '/shell':
                            value = bridge.task_env.shell(body['command'])
                        elif self.path == '/mini-shell':
                            value = bridge.task_env.shell(body['command'],
                                timeout=body.get('timeout', 30), env=body.get('env'),
                                cwd=body.get('cwd', '/app'), truncate=False)
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

    def evaluate_request(self, body):
        if int(body.get('limit', -1)) not in (-1, 1):
            raise ValueError('The smoke benchmark contains one task')
        return self.evaluate(body['model_path'], int(body.get('seed', 72332)),
                             max_tokens=int(body.get('max_tokens', 4000)), agent_request=True)

    def evaluate(self, model_path, seed=72332, max_tokens=4000, agent_request=False):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('An evaluation is already running')
        try:
            if agent_request and not self.accept_agent:
                raise RuntimeError('Training evaluation window has closed')
            if not 1 <= max_tokens <= 131072:
                raise ValueError('Invalid max_tokens')
            if self.cancel.is_set():
                raise RuntimeError('Evaluation cancelled when training ended')
            checkpoint = resolve_checkpoint(model_path, self.home)
            directory = self.result / ('eval_' + secrets.token_hex(6))
            directory.mkdir()
            task = Path(self.suite_task['path']) if self.suite_task else TASK
            base = self.suite_task['base_commit'] if self.suite_task else BASE
            self.task_env = TaskEnvironment(directory, self.images['task'], self.cancel, task=task, base=base)
            config = dict(defaults(task / 'task.toml'), max_tokens=max_tokens,
                          task_id=self.suite_task['id'] if self.suite_task else TASK_ID,
                          task_path=str(task), experiment_mode='full_held_out' if self.suite_task else EXPERIMENT_MODE,
                          held_out_evaluation=bool(self.suite_task))
            # Never silently change the protocol of an already-trained/in-flight run.
            # This file is host-owned, outside the teacher's writable home.
            training_config = self.home.parent / 'run_config.json'
            if training_config.exists():
                trained = json.loads(training_config.read_text())
                config['harness'] = trained.get('harness', 'legacy-json')
                config['enable_thinking'] = trained.get('enable_thinking', False)
                config['model_to_train'] = trained.get('model_to_train', 'Qwen/Qwen3-1.7B-Base')
            if config.get('model_to_train') == 'Qwen/Qwen3.5-9B':
                config['gpu_memory_utilization'] = 0.8
            if config['harness'] == 'mini-swe-agent' and max_tokens >= config['max_model_len']:
                raise ValueError('max_tokens must be smaller than the total context window')
            (directory / 'evaluation_config.json').write_text(json.dumps(config, indent=2))
            self.eval_token = secrets.token_urlsafe(32)
            cache = Path(os.environ['HF_HOME']).resolve()
            cache.mkdir(parents=True, exist_ok=True)
            with cache_overlay(cache, directory / 'cache_overlay') as merged_cache:
                if config['harness'] == 'mini-swe-agent':
                    metrics = self._run_mini_model(checkpoint, directory, merged_cache, seed, config)
                elif config['harness'] == 'legacy-json':
                    metrics = self._run_model(checkpoint, directory, merged_cache, seed)
                else:
                    raise ValueError(f'Unsupported harness: {config["harness"]}')
            metrics['evaluation_directory'] = directory.name
            return metrics
        finally:
            self.eval_token = None
            self.task_env = None
            self.lock.release()

    def _run_model(self, checkpoint, directory, cache, seed):
        cmd = container(self.images['eval'], directory / 'model_home', [
            f'{checkpoint}:/model:ro', f'{ROOT / "src/eval/tasks/deepswe"}:/harness:ro',
            f'{ROOT / "src/eval/per_sample_seed.py"}:/harness_seed.py:ro',
            f'{ROOT / "src/eval/tasks/deepswe_smoke/evaluate_final_eval.py"}:/evaluate_final_eval.py:ro',
            f'{ROOT / "src/eval/templates"}:/templates:ro',
            f'{TASK / "instruction.md"}:/instruction.md:ro',
            f'{directory}:/output', f'{cache}:/hf_cache'], network=True, gpu=True,
            env={'DEEPSWE_WORKER_URL': self.url, 'DEEPSWE_WORKER_TOKEN': self.eval_token,
                 'HF_HOME': '/hf_cache', 'VLLM_API_KEY': 'inspectai', 'PYTHONNOUSERSITE': '1'})
        with (directory / 'evaluate.log').open('w') as log:
            run(cmd + ['python', '/evaluate_final_eval.py', '--seed', str(seed)],
                timeout=EVAL_CONFIG['agent_timeout_seconds'] + EVAL_CONFIG['verifier_timeout_seconds'] + 900,
                stdout=log, cancel=self.cancel)
        metrics = json.loads((directory / 'metrics.json').read_text())
        return metrics

    def _run_mini_model(self, checkpoint, directory, cache, seed, config):
        """Native mini-swe agent on the host; fixed ML SIF serves only local vLLM."""
        python = Path(os.environ['DEEPSWE_MINI_PYTHON'])
        if not python.is_file():
            raise RuntimeError('Run cluster/ibex/deepswe/prepare_mini.sh before submitting')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        api_key = secrets.token_urlsafe(32)
        cmd = container(self.images['eval'], directory / 'model_home', [
            f'{checkpoint}:/model:ro', f'{ROOT / "src/eval/templates"}:/templates:ro',
            f'{cache}:/hf_cache'], network=True, gpu=True,
            env={'HF_HOME': '/hf_cache', 'VLLM_API_KEY': api_key, 'PYTHONNOUSERSITE': '1'})
        cmd += ['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', '/model',
            '--served-model-name', 'deepswe-qwen', '--host', '127.0.0.1', '--port', str(port),
            '--enable-auto-tool-choice', '--max-model-len', str(config['max_model_len']),
            # vLLM 0.11 clamps request max_tokens to the model generation config.
            # Enforce the evaluation budget while retaining all other defaults.
            '--override-generation-config', json.dumps({'max_new_tokens': config['max_tokens']}),
            '--gpu-memory-utilization', str(config['gpu_memory_utilization'])]
        cmd += serving_args(config.get('model_to_train', 'Qwen/Qwen3-1.7B-Base'), ROOT / 'src/eval/templates',
                            enable_thinking=config.get('enable_thinking', True))
        base_url = f'http://127.0.0.1:{port}/v1'
        with (directory / 'vllm.log').open('w') as server_log:
            proc = subprocess.Popen(cmd, env=clean_env(), stdout=server_log,
                stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 600
                while True:
                    if self.cancel.is_set():
                        raise RuntimeError('Evaluation cancelled when training ended')
                    if proc.poll() is not None:
                        raise RuntimeError('vLLM exited during startup; see vllm.log')
                    try:
                        req = urllib.request.Request(base_url + '/models',
                            headers={'Authorization': 'Bearer ' + api_key})
                        with urllib.request.urlopen(req, timeout=2):
                            break
                    except (OSError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('vLLM readiness timeout; see vllm.log')
                        self.cancel.wait(.5)
                # No teacher/cloud credentials or host configuration enter the agent.
                env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                    'HOME': str(directory / 'model_home'), 'LANG': 'C.UTF-8',
                    'PYTHONNOUSERSITE': '1', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
                    'DEEPSWE_WORKER_URL': self.url, 'DEEPSWE_WORKER_TOKEN': self.eval_token,
                    'DEEPSWE_MODEL_URL': base_url, 'DEEPSWE_MODEL_KEY': api_key}
                with (directory / 'evaluate.log').open('w') as log:
                    run([str(python), str(ROOT / 'src/eval/tasks/deepswe/mini_eval.py'),
                        '--output', str(directory), '--seed', str(seed)], env=env,
                        timeout=config['agent_timeout_seconds'] + config['verifier_timeout_seconds'] + 60,
                        stdout=log, cancel=self.cancel)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        return json.loads((directory / 'metrics.json').read_text())


    def close_training_window(self):
        self.accept_agent = False
        self.cancel.set()
        if not self.lock.acquire(timeout=60):
            raise RuntimeError('Exploratory evaluation did not stop after training')
        self.lock.release()

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
