"""Pinned Datacurve DeepSWE: all 113 tasks are final-test-only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

REVISION = '0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea'
TASK_COUNT = 113


def manifest(source: Path) -> dict:
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected DeepSWE revision {REVISION}, got {revision}')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Upstream task checkout must be unmodified')
    tasks = []
    for file in sorted((source / 'tasks').glob('*/task.toml')):
        data = tomllib.loads(file.read_text())
        if data['verifier']['environment_mode'] != 'separate' or data['verifier']['network_mode'] != 'no-network':
            raise ValueError(f'Unsupported verifier isolation: {file}')
        base = data['metadata']['base_commit_hash']
        collect = data['verifier']['collect']
        if len(collect) != 1 or f'git diff --binary {base} HEAD' not in collect[0]['command']:
            raise ValueError(f'Unsupported collection hook: {file}')
        files = {str(p.relative_to(file.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(file.parent.rglob('*')) if p.is_file()}
        tasks.append(dict(id=file.parent.name, path=str(file.parent.resolve()), base_commit=base,
                          image=data['environment']['docker_image'], files=files))
    if len(tasks) != TASK_COUNT:
        raise ValueError(f'Expected {TASK_COUNT} tasks, got {len(tasks)}')
    return dict(dataset='datacurve-ai/deep-swe', revision=revision, split='test', tasks=tasks)


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data['revision'] != REVISION or data['split'] != 'test' or len(data['tasks']) != TASK_COUNT:
        raise ValueError('Invalid full-suite manifest')
    if len({t['id'] for t in data['tasks']}) != TASK_COUNT:
        raise ValueError('Duplicate test tasks')
    return data


def verify_task(task: dict) -> None:
    root = Path(task['path'])
    for name, digest in task['files'].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Task changed since preparation: {name}')


def aggregate(run: Path, suite: dict) -> dict:
    rows = []
    for task in suite['tasks']:
        path = run / 'final_eval' / task['id'] / 'metrics.json'
        if not path.is_file():
            raise RuntimeError(f'Missing final evaluation: {task["id"]}; do not report a partial full-suite score')
        row = json.loads(path.read_text())
        if row['task_id'] != task['id'] or row['accuracy'] not in (0, 1):
            raise ValueError(f'Invalid task result: {path}')
        rows.append(row)
    return dict(accuracy=sum(r['accuracy'] for r in rows) / len(rows), num_tasks=len(rows),
                num_tasks_completed=len(rows), num_tasks_resolved=sum(r['accuracy'] for r in rows), per_task=rows,
                dataset_revision=suite['revision'], held_out_evaluation=True,
                training_task_exposed=False, reference_solution_exposed=False,
                verifier_tests_exposed=False, experiment_mode='full_held_out', seed=72332)
