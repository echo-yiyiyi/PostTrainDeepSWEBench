"""No follow-up submission on failure; exactly one submission after validation."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sequential', ROOT / 'cluster/ibex/deepswe/sequential_watch.py')
sequential = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sequential)


class SequenceTests(unittest.TestCase):
    def run_case(self, state='COMPLETED', code='0:0', marker='123', restart=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images'; images.mkdir()
            (images / 'qwen35_v032_ready.json').write_text(json.dumps({'validation_job_id':marker}))
            result = root / 'results/deepswe_smoke/456';result.mkdir(parents=True)
            (result / 'status.json').write_text('{"status":"completed","judges_complete":true}')
            (result / 'metrics.json').write_text('{"accuracy":0}')
            calls=[]
            def run(cmd, **kwargs):
                calls.append(cmd)
                if cmd[0]=='sacct':
                    job=cmd[cmd.index('-j')+1]
                    return subprocess.CompletedProcess(cmd,0, f'{job}|{state if job=="123" else "COMPLETED"}|{code if job=="123" else "0:0"}\n','')
                if cmd[0]=='bash':kwargs['stdout'].write('Submitted batch job 456\n')
                return subprocess.CompletedProcess(cmd,0,'','')
            directory=root/'logs/deepswe_sequence_123'
            if restart:
                directory.mkdir(parents=True)
                (directory/'status.json').write_text('{"stage":"submitting_smoke"}')
            with contextlib.chdir(root), patch.dict(os.environ,DEEPSWE_STORAGE_IMAGES=str(images),DEEPSWE_STORAGE_RESULTS=str(root/'results')), patch.object(sys,'argv',['watch','123']), patch.object(sequential.subprocess,'run',side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                try:sequential.main()
                except RuntimeError:pass
            return json.loads((directory/'status.json').read_text()), [c for c in calls if c[0]=='bash']

    def test_failures_never_submit(self):
        for state,code in [('FAILED','1:0'),('TIMEOUT','0:0'),('CANCELLED','0:0'),('COMPLETED','1:0')]:
            with self.subTest(state=state):
                data,submitted=self.run_case(state,code)
                self.assertEqual(data['stage'],'stopped');self.assertFalse(submitted)

    def test_matching_success_submits_exactly_once(self):
        data,submitted=self.run_case()
        self.assertEqual(data['stage'],'completed');self.assertEqual(len(submitted),1)
        self.assertEqual(data['smoke_job'],'456')

    def test_wrong_marker_or_uncertain_restart_never_submits(self):
        for kwargs in ({'marker':'999'},{'restart':True}):
            data,submitted=self.run_case(**kwargs)
            self.assertEqual(data['stage'],'stopped');self.assertFalse(submitted)


if __name__=='__main__':unittest.main()
