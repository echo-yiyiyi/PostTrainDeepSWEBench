"""Evaluate an existing smoke-run checkpoint without invoking the training agent."""
import argparse
import json
import os
from pathlib import Path
import time

from runtime import Bridge, resolve_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_run', type=Path)
    args = parser.parse_args()
    source = args.source_run.resolve()
    if not source.is_relative_to(Path(os.environ['DEEPSWE_RESULTS']).resolve()):
        raise ValueError('Source run must be inside DEEPSWE_RESULTS')
    home = source / 'training_home'
    model = '/home/ben/task/final_model'
    checkpoint = resolve_checkpoint(model, home)
    result = source / ('reeval_' + os.environ['SLURM_JOB_ID'])
    result.mkdir(exist_ok=False)
    images = {name: Path(os.environ[f'DEEPSWE_{name.upper()}_SIF']).resolve()
              for name in ('eval', 'task')}
    (result / 'run_config.json').write_text(json.dumps({
        'source_run': str(source), 'checkpoint': str(checkpoint), 'seed': 72332,
        'evaluation_only': True, 'task_fakeroot': True, 'smoke_only': True,
        'images': {key: str(value) for key, value in images.items()}}, indent=2))
    bridge = Bridge(home, result, images)
    bridge.accept_agent = False
    bridge.start()
    started = time.monotonic()
    status = {'status': 'running', 'evaluation_only': True}
    try:
        metrics = bridge.evaluate(model)
        (result / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        status['status'] = 'completed'
        print(json.dumps(metrics, indent=2))
    except Exception as error:
        status.update(status='evaluation_failure', error=str(error))
        raise
    finally:
        status['elapsed_seconds'] = round(time.monotonic() - started, 2)
        (result / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        bridge.server.shutdown()
        bridge.server.server_close()


if __name__ == '__main__':
    main()
