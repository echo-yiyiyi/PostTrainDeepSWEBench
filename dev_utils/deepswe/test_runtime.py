"""No GPU/network dependencies: isolation, authentication and artifact handling."""
import importlib.util
from contextlib import nullcontext
import json
import os
import shutil
import shlex
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval/tasks/deepswe'))
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from runtime import Bridge, TaskEnvironment, container, resolve_checkpoint


class RuntimeTests(unittest.TestCase):
    def test_initialized_repository_can_commit_without_global_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            clean_home = directory / 'clean_home'
            clean_home.mkdir()
            git_env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
            git_env.update(HOME=str(clean_home), XDG_CONFIG_HOME=str(clean_home),
                           GIT_CONFIG_NOSYSTEM='1')

            def git(*args):
                return subprocess.run(['git', '-C', str(directory / 'agent_app'), *args],
                                      env=git_env, text=True, capture_output=True, check=True).stdout

            def extract(_self, target, home):
                git('init')

            def local_setup(cmd, **kwargs):
                # Execute the real setup script against a temporary repository;
                # only replace the container's mount path for this CPU test.
                script = cmd[-1].replace('/app', shlex.quote(str(directory / 'agent_app')))
                subprocess.run(['bash', '-e', '-c', script], env=git_env, check=True)
                return 0, ''

            with patch.object(TaskEnvironment, 'extract', extract), patch('runtime.run', side_effect=local_setup):
                task = TaskEnvironment(directory, 'unused.sif')
            (task.app / 'code.py').write_text('print("base")\n')
            git('add', 'code.py')
            git('commit', '-m', 'base')
            base = git('rev-parse', 'HEAD').strip()
            (task.app / 'code.py').write_text('print("modified")\n')
            git('add', 'code.py')
            git('commit', '-m', 'model change')
            self.assertIn('+print("modified")', git('diff', '--binary', base, 'HEAD'))
            self.assertEqual(git('log', '-1', '--format=%an <%ae>').strip(),
                             'DeepSWE Agent <agent@deepswe.local>')
            self.assertFalse((clean_home / '.gitconfig').exists())

    def test_harness_selection_uses_host_metadata_not_teacher_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / 'training_home'
            checkpoint = home / 'task/final_model'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'config.json').write_text('{}')
            result = root / 'results'
            result.mkdir()
            host_config = root / 'run_config.json'
            host_config.write_text('{}')  # old run: keep its legacy protocol
            agent_config = home / 'task/evaluation_config.json'
            agent_config.write_text('{"harness":"mini-swe-agent"}')
            bridge = Bridge(home, result, {'task': 'unused.sif'})
            try:
                with patch.dict(os.environ, HF_HOME=str(root / 'cache')), patch('runtime.TaskEnvironment'), patch('runtime.cache_overlay', side_effect=lambda *a: nullcontext(root / 'cache')), patch.object(bridge, '_run_model', return_value={'accuracy': 0}) as legacy, patch.object(bridge, '_run_mini_model', return_value={'accuracy': 0}) as native:
                    bridge.evaluate('/home/ben/task/final_model')
                    self.assertEqual(legacy.call_count, 1)
                    native.assert_not_called()
                    host_config.write_text('{"harness":"mini-swe-agent"}')
                    agent_config.write_text('{"harness":"legacy-json"}')
                    bridge.evaluate('/home/ben/task/final_model')
                    self.assertEqual(legacy.call_count, 1)
                    self.assertEqual(native.call_count, 1)
            finally:
                bridge.server.server_close()

    def test_config_is_checkout_local_and_rejects_external_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'
            scripts = root / 'cluster/ibex/deepswe'
            scripts.mkdir(parents=True)
            shutil.copy2(ROOT / 'cluster/ibex/deepswe/common.sh', scripts / 'common.sh')
            sibling = Path(tmp) / 'PostTrainBench'
            sibling.mkdir()
            (sibling / '.env').write_text('OPENAI_API_KEY=must_not_be_read\n')
            env = {k:v for k,v in os.environ.items()
                   if not k.startswith('DEEPSWE_') and k not in ('HF_HOME', 'OPENAI_API_KEY')}
            command = 'module() { :; }; source "$1" || exit; test -z "${OPENAI_API_KEY:-}"; printf "%s" "$DEEPSWE_TRAIN_SIF"'
            result = subprocess.run(['bash', '-c', command, 'test', str(scripts / 'common.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, str(root / 'containers/standard.sif'))
            env['DEEPSWE_TRAIN_SIF'] = str(sibling / 'containers/standard.sif')
            result = subprocess.run(['bash', '-c', command, 'test', str(scripts / 'common.sh')],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must be inside', result.stderr)

    def test_explicit_external_storage_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'project'
            scripts = root / 'cluster/ibex/deepswe'
            scripts.mkdir(parents=True)
            shutil.copy2(ROOT / 'cluster/ibex/deepswe/common.sh', scripts / 'common.sh')
            images = Path(tmp) / 'datasets'
            images.mkdir()
            results = Path(tmp) / 'results'
            (root / '.env').write_text(
                f'DEEPSWE_STORAGE_IMAGES="{images}"\n'
                f'DEEPSWE_STORAGE_RESULTS="{results}"\n')
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('DEEPSWE_') and k not in
                   ('HF_HOME', 'SINGULARITY_CACHEDIR', 'OPENAI_API_KEY')}
            command = ('module() { :; }; source "$1" || exit; '
                       'printf "%s\\n" "$DEEPSWE_TRAIN_SIF" "$HF_HOME" "$DEEPSWE_RESULTS"')
            args = ['bash', '-c', command, 'test', str(scripts / 'common.sh')]
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(images / 'standard.sif'),
                str(results / '.cache/huggingface'), str(results / 'deepswe_smoke')])
            (images / 'standard.sif').symlink_to(Path(tmp) / 'PostTrainBench/standard.sif')
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_checkpoint_scope_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            checkpoint = home / 'task/final_model'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'config.json').write_text('{}')
            self.assertEqual(resolve_checkpoint('/home/ben/task/final_model', home), checkpoint)
            for path in ('/etc', '/home/ben/task/../../elsewhere'):
                with self.assertRaises(ValueError):
                    resolve_checkpoint(path, home)
            (checkpoint / 'weights').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                resolve_checkpoint('/home/ben/task/final_model', home)

    def test_task_container_has_no_network_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = container('task.sif', Path(tmp), ['/tmp/app:/app'])
            self.assertIn('--containall', cmd)
            self.assertIn('--fakeroot', cmd)
            self.assertIn('--cleanenv', cmd)
            self.assertEqual(cmd[cmd.index('--network') + 1], 'none')
            self.assertNotIn('--nv', cmd)
            self.assertNotIn('/tests', ' '.join(cmd))

    def test_gpu_containers_also_use_fakeroot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = container('model.sif', Path(tmp), network=True, gpu=True)
            self.assertIn('--fakeroot', cmd)
            self.assertIn('--nv', cmd)
            self.assertNotIn('--net', cmd)

    def test_verifier_gets_fresh_app_and_readonly_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            calls = []
            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if cmd[-2:] == ['bash', '/tests/test.sh']:
                    (directory / 'verifier/reward.json').write_text('{"reward": 0}')
                return 0, ''
            with patch('runtime.run', side_effect=fake_run):
                env = TaskEnvironment(directory, 'task.sif')
                env.shell('git status')
                reward = env.verify()
            self.assertEqual(reward['reward'], 0)
            shell = next(c for c in calls if c[-1] == 'git status')
            self.assertTrue(all('--fakeroot' in c for c in calls))
            self.assertNotIn('/tests', ' '.join(shell))
            verifier = calls[-1]
            self.assertIn(f'{directory}/verifier_app:/app', verifier)
            self.assertTrue(any(x.endswith(':/tests:ro') for x in verifier))
            self.assertIn(f'{directory}/artifacts:/logs/artifacts:ro', verifier)
            collect = next(c for c in calls if '--binary' in c)
            self.assertEqual(collect[-1], 'HEAD')

    def test_shell_timeout_is_an_observation_not_a_fake_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('runtime.run', return_value=(0, '')):
                env = TaskEnvironment(Path(tmp), 'task.sif')
            with patch('runtime.run', side_effect=subprocess.TimeoutExpired('shell', 60)):
                result = env.shell('sleep 100')
            self.assertEqual(result['exit_code'], 124)
            self.assertNotIn('reward', result)

    def test_bridge_rejects_unauthenticated_and_agent_shell_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = Bridge(Path(tmp), Path(tmp), {})
            bridge.start()
            try:
                for token in ('wrong-token', bridge.agent_token):
                    req = urllib.request.Request(bridge.url + '/shell', data=b'{"command":"id"}',
                        headers={'Authorization': 'Bearer ' + token})
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(req)
                    self.assertEqual(caught.exception.code, 403)
            finally:
                bridge.server.shutdown()
                bridge.server.server_close()


if __name__ == '__main__':
    unittest.main()
