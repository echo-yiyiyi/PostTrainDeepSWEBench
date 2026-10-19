"""Training-time full-suite score interface; test material stays host-only."""
import json
from pathlib import Path
import secrets
import time
from runtime import Bridge, resolve_checkpoint
from full_evaluate import evaluate_task


class TrialBridge(Bridge):
    def __init__(self, home, result, images, suite):
        super().__init__(home, result, images)
        self.suite = suite

    def evaluate_request(self, body):
        limit = int(body.get('limit', -1))
        if limit != -1 and not 1 <= limit <= len(self.suite['tasks']):
            raise ValueError('Invalid task limit')
        max_tokens = int(body.get('max_tokens', 4000))
        if not 1 <= max_tokens < 32768:
            raise ValueError('max_tokens must be between 1 and 32767')
        model_path = body['model_path']
        # Validate agent-controlled paths before creating any host-side processes.
        resolve_checkpoint(model_path, self.home)
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('An evaluation is already running')
        try:
            if not self.accept_agent or self.cancel.is_set():
                raise RuntimeError('Training evaluation window has closed')
            selected = self.suite['tasks'] if limit == -1 else self.suite['tasks'][:limit]
            trial = self.result / ('trial_' + secrets.token_hex(6))
            trial.mkdir()
            seed = int(body.get('seed', 72332))
            config = dict(model_path=model_path, seed=seed, max_tokens=max_tokens, limit=limit,
                          task_ids=[t['id'] for t in selected], evaluation_phase='training_trial')
            (trial / 'request.json').write_text(json.dumps(config, indent=2))
            started = time.monotonic()
            rows = []
            try:
                for task in selected:
                    if self.cancel.is_set():
                        raise RuntimeError('Training evaluation window has closed')
                    rows.append(evaluate_task(self.home, trial / task['id'], self.images['eval'], task,
                                              model_path, seed, max_tokens, self.cancel))
                # Only aggregate metrics cross into the training sandbox: no prompts,
                # per-task identities, test output, reference patches or trajectories.
                metrics = dict(accuracy=sum(r['accuracy'] for r in rows) / len(rows), num_tasks=len(rows),
                               num_tasks_completed=len(rows), seed=seed, max_tokens=max_tokens,
                               evaluation_phase='training_trial')
                (trial / 'metrics.json').write_text(json.dumps(metrics, indent=2))
                return metrics
            except Exception as exc:
                (trial / 'error.json').write_text(json.dumps(dict(error=str(exc),
                    elapsed_seconds=time.monotonic() - started, completed=len(rows))))
                if self.cancel.is_set():
                    raise RuntimeError('Evaluation cancelled at the training deadline') from None
                raise RuntimeError('Trial evaluation failed; detailed diagnostics are retained in host logs') from None
        finally:
            self.lock.release()
