"""Evaluate an existing smoke-run checkpoint without invoking the training agent."""
import argparse
import json
import os
from pathlib import Path
import time

from runtime import resolve_checkpoint
from evaluation_config import TASK_ID, EXPERIMENT_MODE
from training_support import final_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_run', type=Path)
    args = parser.parse_args()
    source = args.source_run.resolve()
    if not source.is_relative_to(Path(os.environ['DEEPSWE_RESULTS']).resolve()):
        raise ValueError('Source run must be inside DEEPSWE_RESULTS')
    source_config = json.loads((source / 'run_config.json').read_text())
    if source_config.get('task_id') != TASK_ID:
        raise ValueError('Source run uses a different task; refusing to relabel its evaluation as the currently selected task')
    home = source / 'training_home'
    model = '/home/ben/task/final_model'
    checkpoint = resolve_checkpoint(model, home)
    result = source / ('reeval_' + os.environ['SLURM_JOB_ID'])
    result.mkdir(exist_ok=False)
    images = {name: Path(os.environ[f'DEEPSWE_{name.upper()}_SIF']).resolve()
              for name in ('eval', 'task')}
    (result / 'run_config.json').write_text(json.dumps({
        'source_run': str(source), 'checkpoint': str(checkpoint), 'seed': 72332,
        'task_id': TASK_ID, 'experiment_mode': EXPERIMENT_MODE,
        'training_task_exposed': True, 'reference_solution_exposed': True,
        'verifier_tests_exposed': True, 'held_out_evaluation': False, 'evaluation_only': True, 'container_fakeroot': True, 'smoke_only': True,
        'images': {key: str(value) for key, value in images.items()}}, indent=2))
    started = time.monotonic()
    status = {'status': 'running', 'evaluation_only': True}
    try:
        final_evaluation(result, home)
        metrics = json.loads((result / 'metrics.json').read_text())
        status['status'] = 'completed'
        print(json.dumps(metrics, indent=2))
    except Exception as error:
        status.update(status='evaluation_failure', error=str(error))
        raise
    finally:
        status['elapsed_seconds'] = round(time.monotonic() - started, 2)
        (result / 'status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
