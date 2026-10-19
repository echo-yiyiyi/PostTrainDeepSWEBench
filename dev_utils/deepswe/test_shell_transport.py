"""Real subprocess stream separation; optional real Singularity/native-agent check."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval/tasks/deepswe'))
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from runtime import TaskEnvironment


class ShellTransportTests(unittest.TestCase):
    def test_launcher_logs_separate_command_stderr_and_exit_code_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('runtime.run', return_value=(0, '')):
                env = TaskEnvironment(Path(tmp), 'fake.sif')
            launcher = [sys.executable, '-c', '''import subprocess,sys
print("INFO: launcher startup", file=sys.stderr, flush=True)
p = subprocess.run(sys.argv[1:])
print("INFO: launcher cleanup", file=sys.stderr, flush=True)
sys.exit(p.returncode)
''']
            with patch('runtime.container', return_value=launcher):
                submitted = env.shell('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT', truncate=False)
                self.assertEqual(submitted, {'exit_code': 0, 'output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'})
                result = env.shell("printf '%s\\n' 'literal \"quotes\" $value'; printf 'WARNING: command error\\n' >&2; exit 7", truncate=False)
            self.assertEqual(result['exit_code'], 7)
            self.assertEqual(result['output'], 'literal "quotes" $value\nWARNING: command error\n')
            log = (Path(tmp) / 'singularity-shell.log').read_text()
            self.assertEqual(log.count('INFO: launcher startup'), 2)
            self.assertEqual(log.count('INFO: launcher cleanup'), 2)
            self.assertNotIn('command error', log)


def image_smoke():
    """Run in a CPU Slurm job, using the pinned mini Python; no model API/GPU."""
    import litellm
    import mini_eval
    from runtime import Bridge, EVAL_CONFIG
    from evaluation_config import TASK_ID, EXPERIMENT_MODE
    sys.path.insert(0, str(Path(__file__).parent))
    from test_mini import response

    directory = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('shell_transport_' + os.environ['SLURM_JOB_ID'])
    directory.mkdir()
    task = TaskEnvironment(directory, Path(os.environ['DEEPSWE_TASK_SIF']))
    result = task.shell("printf 'compiler diagnostic\\n' >&2; exit 7", truncate=False)
    assert result == {'exit_code': 7, 'output': 'compiler diagnostic\n'}, result
    bridge = Bridge(directory / 'unused_home', directory, {})
    bridge.task_env = task
    bridge.eval_token = 'local-test-token'
    bridge.accept_agent = False
    bridge.start()
    try:
        with patch.dict(os.environ, DEEPSWE_WORKER_URL=bridge.url,
                DEEPSWE_WORKER_TOKEN=bridge.eval_token,
                DEEPSWE_MODEL_URL='http://127.0.0.1:9/v1', DEEPSWE_MODEL_KEY='test'), patch.object(
                litellm, 'completion', return_value=response('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')) as model:
            outcome = mini_eval.run_agent(dict(EVAL_CONFIG, task_id=TASK_ID,
                experiment_mode=EXPERIMENT_MODE), directory, 72332, 'Submit immediately.')
        assert outcome['exit_status'] == 'Submitted', outcome
        assert model.call_count == 1, model.call_count
        assert (directory / 'singularity-shell.log').stat().st_size > 0
        (directory / 'result.json').write_text(json.dumps({
            'passed': True, 'native_exit_status': outcome['exit_status'],
            'model_calls': model.call_count, 'command_stderr_preserved': True,
            'launcher_diagnostics_separated': True}, indent=2))
        print('PASS: native mini-swe submits after ONE call in the real Singularity task image; command stderr and exit code preserved.')
    finally:
        bridge.server.shutdown()
        bridge.server.server_close()


if __name__ == '__main__':
    if '--image-smoke' in sys.argv:
        image_smoke()
    else:
        unittest.main()
