"""Submit smoke only AFTER a successful environment job and verified artifacts.

Run detached via sequential_watch.sh. Unknown accounting states never authorize
submission; failed/cancelled/timed-out jobs stop the chain. A per-run lock and
persisted submission intent prevent duplicate submissions on restart.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time


def job_state(job):
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', job,
                             '--format=JobIDRaw,State,ExitCode'], capture_output=True, text=True, check=True)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.split('|')[0] == job]
    if len(rows) != 1:
        return 'UNKNOWN', ''
    return rows[0][1].split()[0].rstrip('+'), rows[0][2]


def successful(state, code):
    return state == 'COMPLETED' and code == '0:0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_job')
    parser.add_argument('--smoke-walltime', default='06:00:00')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+', args.environment_job):
        raise ValueError('Invalid job ID')
    directory = Path('logs') / ('deepswe_sequence_' + args.environment_job)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = directory / 'status.json'
    data = json.loads(path.read_text()) if path.exists() else dict(environment_job=args.environment_job, stage='waiting_environment')

    def save(**fields):
        data.update(fields, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        tmp.replace(path)
        print(json.dumps(data), flush=True)

    def wait(job):
        while True:
            try:
                state, code = job_state(job)
            except subprocess.CalledProcessError as exc:
                save(accounting_error=str(exc))
                time.sleep(30)
                continue
            save(current_job=job, slurm_state=state, exit_code=code)
            if successful(state, code):
                return True
            if state in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED'} or (state == 'COMPLETED' and code != '0:0'):
                save(stage='stopped', reason=f'Job {job}: {state} {code}; no subsequent submission')
                return False
            time.sleep(30)

    try:
        if data['stage'] in ('stopped', 'completed'):
            return
        if data['stage'] == 'submitting_smoke':
            raise RuntimeError('Prior submission outcome uncertain; refusing automatic resubmission')
        if not data.get('smoke_job'):
            if not wait(args.environment_job):
                return
            ready = json.loads((Path(os.environ['DEEPSWE_STORAGE_IMAGES']) / 'qwen35_v032_ready.json').read_text())
            if ready.get('validation_job_id') != args.environment_job:
                raise RuntimeError('Ready marker does not match this successful environment job')
            subprocess.run(['python3', 'cluster/ibex/deepswe/check_full_ready.py'], check=True)
            save(stage='submitting_smoke')
            with (directory / 'submission.log').open('w') as log:
                result = subprocess.run(['bash','src/commit_utils/commit_deepswe_smoke.sh','--walltime',args.smoke_walltime], stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Smoke submission failed; see submission.log')
            matches = re.findall(r'^Submitted batch job (\d+)$', (directory / 'submission.log').read_text(), re.M)
            if len(matches) != 1:
                raise RuntimeError('Uncertain smoke submission outcome; do not resubmit automatically')
            save(stage='waiting_smoke', smoke_job=matches[0])
        if not wait(data['smoke_job']):
            return
        result_dir = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / 'deepswe_smoke' / data['smoke_job']
        status = json.loads((result_dir / 'status.json').read_text())
        metrics = json.loads((result_dir / 'metrics.json').read_text())
        if status.get('status') != 'completed' or not status.get('judges_complete'):
            raise RuntimeError('Smoke scheduler completed but run/judge artifacts are incomplete')
        save(stage='completed', result_dir=str(result_dir), accuracy=metrics['accuracy'])
    except Exception as exc:
        save(stage='stopped', reason=str(exc))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
