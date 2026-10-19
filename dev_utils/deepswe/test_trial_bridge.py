"""Training trials: real HTTP client, private outputs and deadline cancellation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import tomllib
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/deepswe'))
from trial_bridge import TrialBridge
from run import prepare_training_home


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.home = root / 'home'
        (self.home / 'task/final_model').mkdir(parents=True)
        (self.home / 'task/final_model/config.json').write_text('{}')
        self.result = root / 'result'
        self.result.mkdir()
        self.bridge = TrialBridge(self.home, self.result, {'eval':Path('eval.sif')},
            {'tasks':[{'id':'private-a'}, {'id':'private-b'}]})
        self.addCleanup(self.bridge.server.server_close)
        self.body = {'model_path':'/home/ben/task/final_model', 'limit':1, 'seed':22, 'max_tokens':3000}

    def test_client_http_round_trip_only_aggregate_returned(self):
        self.bridge.start()
        try:
            with patch('trial_bridge.evaluate_task', return_value={
                    'accuracy':1, 'task_id':'private-a', 'question':'SECRET', 'reward':{'private_test':'SECRET'}}) as evaluate:
                output = self.home / 'metrics.json'
                env = dict(os.environ, DEEPSWE_EVAL_URL=self.bridge.url, DEEPSWE_EVAL_TOKEN=self.bridge.agent_token)
                proc = subprocess.run([sys.executable, str(ROOT / 'src/eval/tasks/deepswe/evaluate.py'),
                    '--model-path', '/home/ben/task/final_model', '--limit', '1', '--seed', '22',
                    '--max-tokens', '3000', '--templates-dir', 'templates', '--json-output-file', str(output)],
                    env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                metrics = json.loads(output.read_text())
                self.assertEqual(metrics['accuracy'],1)
                self.assertEqual(metrics['num_tasks'],1)
                self.assertNotIn('SECRET', output.read_text())
                self.assertNotIn('private-a', proc.stdout)
                self.assertEqual(evaluate.call_args.args[4:7], ('/home/ben/task/final_model',22,3000))
        finally:
            self.bridge.server.shutdown()

    def test_all_tasks_and_no_silent_partial_score(self):
        with patch('trial_bridge.evaluate_task', side_effect=[{'accuracy':1}, {'accuracy':0}]):
            result = self.bridge.evaluate_request(dict(self.body, limit=-1))
        self.assertEqual(result['accuracy'],0.5)
        with patch('trial_bridge.evaluate_task', side_effect=[{'accuracy':1}, RuntimeError('SECRET TEST CONTENT')]):
            with self.assertRaisesRegex(RuntimeError, '^Trial evaluation failed; detailed diagnostics are retained in host logs$'):
                self.bridge.evaluate_request(dict(self.body, limit=-1))
        self.assertEqual(len(list(self.result.glob('trial_*/metrics.json'))),1)

    def test_untrusted_paths_and_limits_never_start_evaluation(self):
        with patch('trial_bridge.evaluate_task') as evaluate:
            for body in [dict(self.body,model_path='/etc'), dict(self.body,limit=0),dict(self.body,max_tokens=32768)]:
                with self.assertRaises(ValueError):
                    self.bridge.evaluate_request(body)
            evaluate.assert_not_called()

    def test_deadline_cancels_active_trial_and_blocks_new_requests(self):
        entered = threading.Event()
        errors = []
        def fake_evaluate(*args):
            entered.set()
            self.assertTrue(args[-1].wait(5))
            raise RuntimeError('cancelled')
        def worker():
            try:
                self.bridge.evaluate_request(self.body)
            except RuntimeError as exc:
                errors.append(str(exc))
        with patch('trial_bridge.evaluate_task', side_effect=fake_evaluate):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(entered.wait(5))
            self.bridge.close_training_window()
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors,['Evaluation cancelled at the training deadline'])
            with self.assertRaisesRegex(RuntimeError,'window has closed'):
                self.bridge.evaluate_request(self.body)

    def test_modern_shell_policy_preserves_image_cuda_environment(self):
        prepare_training_home(self.home, modern=True)
        config = tomllib.loads((self.home / '.codex/config.toml').read_text())
        values = config['shell_environment_policy']['set']
        self.assertTrue(values['PATH'].startswith('/home/ben/.venv/bin:'))
        for name in ('NV_CUDA_LIB_VERSION','NV_LIBNCCL_PACKAGE','LD_LIBRARY_PATH'):
            self.assertNotIn(name,values)


if __name__ == '__main__':
    unittest.main()
