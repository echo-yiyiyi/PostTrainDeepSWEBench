"""Full-suite boundaries, complete-only scoring, and prompt policy regressions."""
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
sys.path.insert(0, str(ROOT / 'src/judges'))
from full_suite import aggregate, load_manifest, verify_task, REVISION, TASK_COUNT
from model_profiles import serving_args, QWEN35_ID
from training_support import make_general_prompt
from get_judge_prompt import fill_api_usage


class FullTests(unittest.TestCase):
    def test_full_prompt_no_test_access_or_blanket_api_ban(self):
        with patch.dict(os.environ, DEEPSWE_MODE='full'):
            prompt = make_general_prompt(QWEN35_ID, 36000)
        self.assertEqual(prompt.strip(), (ROOT / 'src/eval/tasks/deepswe/prompt.preview.txt').read_text().strip())
        self.assertIn('maximum of 10.0 hours', prompt)
        self.assertIn('Codex may act as the synthetic-data pipeline agent', prompt)
        self.assertIn('all 113 Datacurve DeepSWE tasks are held-out tests', prompt)
        self.assertIn('You can query the benchmark via the evaluate.py script.', prompt)
        self.assertNotIn('You must NOT use any API key', prompt)
        self.assertNotIn('reference_materials/', prompt)
        self.assertNotIn('Abort pending body reads', prompt)
        self.assertIn('python evaluate.py --model-path final_model', prompt)
        self.assertIn('uv pip install --python /home/ben/.venv/bin/python', prompt)
        self.assertNotIn('only after training', prompt)

    def test_qwen35_smoke_uses_full_policy_without_teaching_material(self):
        with patch.dict(os.environ, DEEPSWE_MODE='smoke'):
            prompt = make_general_prompt(QWEN35_ID, 3600)
        self.assertEqual(prompt.strip(), (ROOT / 'src/eval/tasks/deepswe_smoke/prompt.preview.txt').read_text().strip())
        self.assertIn('maximum of 1.0 hours', prompt)
        self.assertIn('Qwen/Qwen3.5-9B', prompt)
        self.assertIn('Trajectory Judge Agent', prompt)
        self.assertIn('one fixed held-out task', prompt)
        self.assertNotIn('reference_materials/', prompt)
        self.assertNotIn('Abort pending body reads', prompt)
        self.assertNotIn('final evaluation covers all 113', prompt)

    def test_api_judge_same_policy_and_original_unchanged(self):
        template = (ROOT / 'src/judges/api_usage_judge/prompt.md').read_text()
        full = fill_api_usage(template, 'deepswe', 'codex', 'gpt-6-sol')
        self.assertIn('Codex may act as the synthetic-data pipeline agent', full)
        self.assertNotIn('Any such call is a violation', full)
        self.assertIn('Any such call is a violation', fill_api_usage(template, 'gsm8k', 'codex', 'gpt-6-sol'))

    def test_aggregate_requires_every_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            suite = {'revision': REVISION, 'tasks': [{'id':'a'}, {'id':'b'}]}
            a = run / 'final_eval/a'
            a.mkdir(parents=True)
            (a / 'metrics.json').write_text(json.dumps({'task_id':'a', 'accuracy':1}))
            with self.assertRaises(RuntimeError):
                aggregate(run, suite)
            b = run / 'final_eval/b'
            b.mkdir()
            (b / 'metrics.json').write_text(json.dumps({'task_id':'b', 'accuracy':0}))
            metrics = aggregate(run, suite)
            self.assertEqual(metrics['accuracy'], 0.5)
            self.assertTrue(metrics['held_out_evaluation'])
            self.assertFalse(metrics['training_task_exposed'])

    def test_manifest_rejects_subset_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'manifest.json'
            for rows in [[], [{'id':'same'}] * TASK_COUNT]:
                p.write_text(json.dumps({'revision':REVISION, 'split':'test', 'tasks':rows}))
                with self.assertRaises(ValueError):
                    load_manifest(p)

    def test_changed_verifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'test.sh'
            p.write_text('tampered')
            with self.assertRaises(ValueError):
                verify_task({'path':tmp, 'files':{'test.sh':'wrong-hash'}})

    def test_qwen35_protocol_is_not_qwen3(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'qwen3_5.jinja').touch()
            args = serving_args(QWEN35_ID, p)
            self.assertIn('qwen3_coder', args)
            self.assertIn('--language-model-only', args)
            self.assertNotIn('hermes', args)
            kwargs = json.loads(args[args.index('--default-chat-template-kwargs') + 1])
            self.assertIs(kwargs['enable_thinking'], True)
            old = serving_args(QWEN35_ID, p, enable_thinking=False)
            self.assertIs(json.loads(old[old.index('--default-chat-template-kwargs') + 1])
                          ['enable_thinking'], False)

    def test_training_eval_client_help_and_limits(self):
        client = str(ROOT / 'src/eval/tasks/deepswe/evaluate.py')
        result = subprocess.run([sys.executable, client, '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--limit', result.stdout)
        for limit in ('0', '114'):
            result = subprocess.run([sys.executable, client, '--limit', limit], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('--limit must be', result.stderr)


if __name__ == '__main__':
    unittest.main()
