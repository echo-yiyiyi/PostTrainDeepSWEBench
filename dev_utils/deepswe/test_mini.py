"""Exercise the real pinned mini-swe loop without GPU or external model calls."""
import importlib.resources
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/deepswe'))
import mini_eval
from evaluation_config import defaults, TASK_DIRECTORY
from litellm import ModelResponse
import litellm


def response(command=None):
    message = {'role': 'assistant', 'content': 'Running the next command.'}
    if command is not None:
        message['tool_calls'] = [{'id': 'call_test', 'type': 'function',
            'function': {'name': 'bash', 'arguments': json.dumps({'command': command})}}]
    return ModelResponse(model='deepswe-qwen', choices=[{'index': 0,
        'message': message, 'finish_reason': 'tool_calls' if command else 'stop'}],
        usage={'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30})


class MiniTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.config = defaults(ROOT / 'src/eval/tasks/deepswe_smoke' / TASK_DIRECTORY / 'task.toml')
        self.config.update(task_id='test', experiment_mode='test')
        self.env = patch.dict(os.environ, DEEPSWE_MODEL_URL='http://127.0.0.1:9/v1',
                              DEEPSWE_MODEL_KEY='temporary-test-key')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.commands = []

    def rpc(self, operation, body):
        if operation == '/verify':
            return {'reward': 0}
        command = body['command']
        self.commands.append(command)
        if command.startswith('python3 -c'):
            return {'exit_code': 0, 'output': json.dumps(dict(system='Linux', release='test',
                version='test', machine='x86_64', node='task', processor=''))}
        if command == 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT':
            return {'exit_code': 0, 'output': 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'}
        return {'exit_code': 0, 'output': 'clean\n'}

    def test_native_submit_max_tokens_and_distinct_seeds(self):
        with patch.object(mini_eval, 'rpc', self.rpc), patch.object(litellm, 'completion',
                side_effect=[response('git status'), response('echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')]) as api:
            result = mini_eval.run_agent(self.config, self.directory, 72332, 'Fix the repository')
        self.assertEqual(result['exit_status'], 'Submitted')
        calls = [c.kwargs for c in api.call_args_list]
        self.assertEqual([c['max_tokens'] for c in calls], [4000, 4000])
        self.assertNotEqual(calls[0]['seed'], calls[1]['seed'])
        self.assertEqual(calls[0]['tools'][0]['function']['name'], 'bash')
        self.assertEqual(self.commands[-2:], ['git status', 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'])
        self.assertNotIn('temporary-test-key', (self.directory / 'mini-swe-agent.trajectory.json').read_text())

    def test_context_exhaustion_stops_after_one_call(self):
        error = litellm.exceptions.ContextWindowExceededError('maximum context length', model='test', llm_provider='openai')
        with patch.object(mini_eval, 'rpc', self.rpc), patch.object(litellm, 'completion', side_effect=error) as api:
            result = mini_eval.run_agent(self.config, self.directory, 1, 'Fix the repository')
        self.assertEqual(result['exit_status'], 'ContextWindowExceededError')
        self.assertEqual(api.call_count, 1)
        self.assertEqual(len(self.commands), 1)  # only system information

    def test_native_format_error_limit(self):
        with patch.object(mini_eval, 'rpc', self.rpc), patch.object(litellm, 'completion', return_value=response()) as api:
            result = mini_eval.run_agent(self.config, self.directory, 1, 'Fix the repository')
        self.assertEqual(result['exit_status'], 'RepeatedFormatError')
        self.assertEqual(api.call_count, 3)

    def test_auth_failure_is_not_a_model_zero(self):
        error = litellm.exceptions.AuthenticationError('test auth error', model='test', llm_provider='openai')
        with patch.object(mini_eval, 'rpc', self.rpc), patch.object(litellm, 'completion', side_effect=error) as api:
            with self.assertRaises(litellm.exceptions.AuthenticationError):
                mini_eval.run_agent(self.config, self.directory, 1, 'Fix the repository')
        self.assertEqual(api.call_count, 1)
        self.assertFalse((self.directory / 'metrics.json').exists())

    def test_vendored_prompt_matches_pinned_package(self):
        native = importlib.resources.files('minisweagent').joinpath('config/mini.yaml').read_bytes()
        self.assertEqual(native, (ROOT / 'src/deepswe/mini_swe/mini.yaml').read_bytes())
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
        self.assertEqual(BASH_TOOL, json.loads((ROOT / 'src/deepswe/mini_swe/bash_tool.json').read_text()))

    def test_inspect_scores_native_context_exhaustion_once(self):
        (self.directory / 'evaluation_config.json').write_text(json.dumps(self.config))
        error = litellm.exceptions.ContextWindowExceededError('maximum context length', model='test', llm_provider='openai')
        with patch.object(mini_eval, 'rpc', side_effect=self.rpc) as worker, patch.object(litellm, 'completion', side_effect=error), patch.object(sys, 'argv', ['mini_eval.py', '--output', str(self.directory), '--seed', '1']):
            mini_eval.main()
        self.assertEqual(sum(c.args[0] == '/verify' for c in worker.call_args_list), 1)
        metrics = json.loads((self.directory / 'metrics.json').read_text())
        self.assertEqual(metrics['accuracy'], 0)
        self.assertEqual(metrics['agent_exit_status'], 'ContextWindowExceededError')


if __name__ == '__main__':
    unittest.main()
