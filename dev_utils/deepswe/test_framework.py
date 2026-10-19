"""Regression checks for transport and shared orchestration, without model APIs."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/deepswe'))
from runtime import run
from training_support import make_prompt, make_general_prompt, prepare_teaching_materials
from runtime import TASK
from evaluation_config import TEACHING_NAME, TASK_ID, BASE_COMMIT, TASK_IMAGE


class FrameworkTests(unittest.TestCase):
    def test_selected_task_matches_upstream_contract_and_metadata(self):
        contract = (TASK / 'task.toml').read_text()
        self.assertIn(f'task_id = "{TASK_ID}"', contract)
        self.assertIn(f'base_commit_hash = "{BASE_COMMIT}"', contract)
        self.assertIn(f'docker_image = "{TASK_IMAGE}"', contract)
        metadata = json.loads((TASK.parent / 'info.json').read_text())
        self.assertEqual(metadata['task_id'], TASK_ID)
        prompt = make_prompt('Qwen/Qwen3-1.7B-Base', 3600)
        self.assertIn(BASE_COMMIT, prompt)

    def test_patch_stdout_excludes_container_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            patch_file = Path(directory) / 'model.patch'
            for content in ('', 'diff --git a/a b/a\n'):
                with patch_file.open('w') as stdout, (Path(directory) / 'collect.log').open('w') as stderr:
                    run([sys.executable, '-c',
                         'import sys;sys.stdout.write(sys.argv[1]);sys.stderr.write("WARNING: container")', content],
                        stdout=stdout, stderr=stderr)
                self.assertEqual(patch_file.read_text(), content)

    def test_training_end_cancels_evaluation(self):
        cancel = threading.Event()
        timer = threading.Timer(.1, cancel.set)
        timer.start()
        try:
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                run([sys.executable, '-c', 'import time;time.sleep(30)'], cancel=cancel)
        finally:
            timer.join()

    def test_shared_prompt_keeps_rules_and_ibex_install_path(self):
        prompt = make_general_prompt('Qwen/Qwen3-1.7B-Base', 3600)
        self.assertIn('Do not modify evaluate.py or templates/', prompt)
        self.assertIn('must NOT use any API key', prompt)
        self.assertIn('one held-out repository task', prompt)
        self.assertIn('uv pip install --python', prompt)
        self.assertNotIn('H100', prompt)
        self.assertNotIn('pip install --system', prompt)

    def test_teaching_prompt_and_reference_scope(self):
        prompt = make_prompt('Qwen/Qwen3-1.7B-Base', 3600)
        self.assertIn('<evaluation_task>\n' + (TASK / 'instruction.md').read_text() + '</evaluation_task>', prompt)
        self.assertIn('Training budget: 60 minutes', prompt)
        self.assertIn('has 90 minutes', prompt)
        self.assertIn('official reference solution and verifier tests', prompt)
        self.assertIn('NOT available to that model at evaluation time', prompt)
        self.assertNotIn('{base_model}', prompt)
        self.assertEqual(prompt, (ROOT/'src/deepswe/prompts'/f'teach_{TEACHING_NAME}.rendered.txt').read_text())
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory)
            prepare_teaching_materials(task, mode='general')
            self.assertFalse((task/'reference_materials').exists())
            prepare_teaching_materials(task)
            for name in ('test.patch', 'test.sh', 'grader.py', 'config.json'):
                self.assertEqual((task/'reference_materials/tests'/name).read_bytes(), (TASK/'tests'/name).read_bytes())
            self.assertEqual((task/'reference_materials/solution/solution.patch').read_bytes(),
                (ROOT/'src/deepswe/teaching_materials'/TEACHING_NAME/'solution/solution.patch').read_bytes())

    def test_http_error_keeps_diagnostic(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('client', ROOT / 'src/eval/tasks/deepswe_smoke/evaluate.py')
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        error = urllib.error.HTTPError('http://localhost', 500, 'failure', {}, io.BytesIO(b'{"error":"checkpoint invalid"}'))
        with patch.dict(os.environ, DEEPSWE_EVAL_URL='http://localhost', DEEPSWE_EVAL_TOKEN='test'), patch.object(sys, 'argv', ['evaluate.py']), patch.object(client.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'checkpoint invalid'):
                client.main()

    def test_all_shared_judge_slots_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'training_home/task'
            task.mkdir(parents=True)
            (root / 'training_home/test_data.json').write_text('[]')
            (root / 'solve_parsed.txt').write_text('Synthetic framework test')
            (root / 'run_config.json').write_text(json.dumps({'model_to_train': 'Qwen/Qwen3-1.7B-Base'}))
            shim = root / 'bin'
            shim.mkdir()
            # Intercept only the paid model launcher. Run the real prompt builders,
            # trace parsers, shared slot orchestration and verdict collection.
            launcher = shim / 'python3'
            import shlex
            launcher.write_text('#!/bin/bash\n'
                'if [[ "$1" == src/deepswe/judge_exec.py ]]; then\n'
                '  printf \'{"mock_framework_test":true}\\n\' > "$2/task/judgement.json"\n'
                '  : > "$3"\n'
                '  exit 0\n'
                'fi\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
            launcher.chmod(0o755)
            proc = subprocess.run(['bash', 'src/deepswe/judge.sh', str(root)], cwd=ROOT,
                env=dict(os.environ, PATH=str(shim)+os.pathsep+os.environ['PATH']),
                capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertTrue(json.loads((root/'judges_status.json').read_text())['complete'])
            self.assertEqual(len(list(root.glob('judgement_*.json'))) +
                             len(list((root/'judgement_multi_runs').glob('judgement_*.json'))), 6)

    def test_shared_final_cascade_accepts_zero_without_retry(self):
        # Only replace the costly model invocation; execute the real shell cascade
        # and metrics aggregation, with no GPU commands or container launches.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('src/eval/tasks/deepswe_smoke', 'src/deepswe', 'src/utils'):
                (root / name).mkdir(parents=True, exist_ok=True)
            import shutil
            for name in ('src/eval/run_final_eval.sh', 'src/utils/aggregate_seed_metrics.py', 'src/eval/tasks/deepswe_smoke/evaluate_final_eval.py'):
                shutil.copy2(ROOT / name, root / name)
            (root / 'src/deepswe/final_attempt.py').write_text('''import sys,json
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--json-output-file')+1])
with (p.parent/'attempts').open('a') as f: f.write('attempt\\n')
p.write_text(json.dumps({'accuracy':0}))
''')
            proc = subprocess.run(['bash', 'src/eval/run_final_eval.sh', 'deepswe_smoke', str(root/'model'), str(root/'out'), str(root/'metrics.json')],
                cwd=root, env=dict(os.environ, EVAL_RUNTIME='deepswe_smoke'), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual((root/'out/attempts').read_text(), 'attempt\n')
            self.assertEqual(json.loads((root/'metrics.json').read_text())['accuracy'], 0)


if __name__ == '__main__':
    unittest.main()
