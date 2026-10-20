"""Real task-image preflight. Run on an Ibex compute node, without a GPU."""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval/tasks/deepswe'))
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from runtime import BASE, TaskEnvironment, container, run


with tempfile.TemporaryDirectory(prefix='task-fakeroot-', dir=ROOT / 'results') as tmp:
    directory = Path(tmp)
    task = TaskEnvironment(directory, Path(os.environ['DEEPSWE_TASK_SIF']))
    print('PASS: complete /app copied, including node_modules', flush=True)
    check = task.shell(f'''set -e
test "$(id -u)" = 0
test -r /app/package.json
test ! -e /tests
test -z "${{OPENAI_API_KEY:-}}${{CODEX_API_KEY:-}}${{DEEPSWE_EVAL_TOKEN:-}}"
test "$(git rev-parse HEAD)" = {BASE}
git diff --exit-code HEAD
python3 -c 'import pathlib; routes=pathlib.Path("/proc/net/route").read_text().splitlines()[1:]; assert not any(line.split()[1] == "00000000" for line in routes)'
printf persistent > /app/.deepswe_probe
npx vitest run test/unit.test.ts --coverage=false
''')
    assert check['exit_code'] == 0, check
    print(check['output'], flush=True)
    check = task.shell('test "$(cat /app/.deepswe_probe)" = persistent')
    assert check['exit_code'] == 0, check
    print('PASS: fakeroot, network isolation, credential isolation and persistent edits', flush=True)
    app = directory / 'verifier_app'
    home = directory / 'verifier_home'
    task.extract(app, home)
    run(container(task.image, home, [f'{app}:/app'], pwd='/app') +
        ['bash', '-c', 'test ! -e .deepswe_probe && test -r package.json && git diff --exit-code HEAD'])
    print('PASS: separate verifier workspace starts pristine with the complete repository', flush=True)
