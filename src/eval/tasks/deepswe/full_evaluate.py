"""Shared host task evaluator and independent final-evaluation CLI."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from full_suite import load_manifest, verify_task, aggregate
from runtime import Bridge, run


def evaluate_task(home, result, eval_image, task, model_path='/home/ben/task/final_model',
                  seed=72332, max_tokens=4000, cancel=None):
    verify_task(task)
    result.mkdir(parents=True, exist_ok=True)
    # Task images stay on the host, outside every training-container mount.
    with tempfile.TemporaryDirectory(prefix='deepswe-task-', dir=os.environ['DEEPSWE_CONTAINER_TMPDIR']) as tmp:
        image = Path(tmp) / 'task.sif'
        with (result / 'image_prepare.log').open('w') as log:
            run(['singularity', 'pull', '--disable-cache', str(image), 'docker://' + task['image']],
                timeout=3600, stdout=log, cancel=cancel)
        bridge = Bridge(home, result, {'eval': eval_image, 'task': image}, suite_task=task)
        bridge.accept_agent = False
        if cancel is not None:
            bridge.cancel = cancel
        bridge.start()
        try:
            metrics = bridge.evaluate(model_path, seed=seed, max_tokens=max_tokens)
        finally:
            bridge.server.shutdown()
            bridge.server.server_close()
        (result / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        # Preserve submitted patches, trajectories and test reports, not dependency trees.
        for attempt in result.glob('eval_*'):
            for name in ('agent_app', 'verifier_app'):
                path = attempt / name
                if path.is_dir():
                    shutil.rmtree(path)
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--seed', type=int, default=72332)
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.seed != 72332:
        raise ValueError('The complete-suite protocol fixes the evaluation seed to 72332')
    source = args.run.resolve()
    suite = load_manifest(source / 'suite_manifest.json')
    config = json.loads((source / 'run_config.json').read_text())
    status = json.loads((source / 'status.json').read_text())
    if config['experiment_mode'] != 'full_held_out' or status['status'] not in ('training_completed', 'completed'):
        raise ValueError('Full-suite evaluation requires completed full-suite training')
    if args.aggregate:
        metrics = aggregate(source, suite)
        (source / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        status.update(status='completed', phase='final_evaluation_completed')
        (source / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps({'accuracy': metrics['accuracy'], 'num_tasks': metrics['num_tasks']}))
        return
    if args.index is None or not 0 <= args.index < len(suite['tasks']):
        raise ValueError('A valid task index is required')
    task = suite['tasks'][args.index]
    verify_task(task)
    result = source / 'final_eval' / task['id']
    result.mkdir(parents=True, exist_ok=True)
    if (result / 'metrics.json').exists():
        raise FileExistsError('Task already scored; preserve previous results')
    started = time.monotonic()
    state = {'status': 'running', 'task_id': task['id']}
    try:
        evaluate_task(source / 'training_home', result, Path(config['images']['eval']), task)
        state['status'] = 'completed'
    except Exception as exc:
        state.update(status='infrastructure_failure', error=str(exc))
        raise
    finally:
        state['elapsed_seconds'] = round(time.monotonic() - started, 2)
        (result / 'status.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
