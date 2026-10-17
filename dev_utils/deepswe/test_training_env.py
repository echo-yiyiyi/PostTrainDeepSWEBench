"""Real non-root SIF check: no GPU, network, credentials or model API calls."""
from pathlib import Path
import sys
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/deepswe'))
from run import prepare_training_home, TRAIN_ENV
from runtime import container, run


with tempfile.TemporaryDirectory(prefix='env-check-', dir=ROOT / 'dev_utils/deepswe') as tmp:
    home = Path(tmp)
    prepare_training_home(home)
    config = tomllib.loads((home / '.codex/config.toml').read_text())
    policy = config['shell_environment_policy']['set']
    assert all(policy[key] == value for key, value in TRAIN_ENV.items())
    cmd = container(ROOT / 'containers/standard.sif', home, env=TRAIN_ENV)
    _, output = run(cmd + ['bash', '/home/ben/setup_training_env.sh'], timeout=120)
    print(output)
    # A tiny valid wheel exercises installation without downloading packages.
    wheel = home / 'deepswe_env_probe-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('deepswe_env_probe.py', 'VALUE = 42\n')
        archive.writestr('deepswe_env_probe-1.0.dist-info/METADATA',
                         'Metadata-Version: 2.1\nName: deepswe-env-probe\nVersion: 1.0\n')
        archive.writestr('deepswe_env_probe-1.0.dist-info/WHEEL',
                         'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr('deepswe_env_probe-1.0.dist-info/RECORD', '')
    (home / 'check.py').write_text('''import os, sys, torch, transformers, deepswe_env_probe
assert os.getuid() != 0
assert sys.executable == '/home/ben/.venv/bin/python', sys.executable
assert deepswe_env_probe.VALUE == 42
assert deepswe_env_probe.__file__.startswith('/home/ben/.venv/')
assert not torch.__file__.startswith('/home/ben/.venv/'), torch.__file__
print('PASS: non-root shell uses venv; torch inherited; extra package installed locally.')
''')
    # Apply the actual generated Codex shell policy, including its PATH overrides.
    shell = container(ROOT / 'containers/standard.sif', home) + ['env'] + [
        f'{key}={value}' for key, value in policy.items()]
    _, output = run(shell + ['bash', '-c',
        'set -e; uv pip install --offline --python /home/ben/.venv/bin/python '
        '/home/ben/deepswe_env_probe-1.0-py3-none-any.whl; python /home/ben/check.py; '
        '/usr/bin/python -c "import importlib.util; '
        'assert importlib.util.find_spec(\'deepswe_env_probe\') is None"'], timeout=120)
    print(output)
    _, output = run(shell + ['bash', '-lc', 'command -v uv; python /home/ben/check.py'], timeout=120)
    print(output)
    # GPU preflight must fail before the agent when no GPU is exposed.
    code, output = run(container(ROOT / 'containers/standard.sif', home,
                                env={**TRAIN_ENV, 'DEEPSWE_REQUIRE_CUDA': '1'}) +
                       ['bash', '/home/ben/setup_training_env.sh'], timeout=120, check=False)
    assert code != 0 and 'Training GPU is unavailable' in output, output
    print('PASS: missing GPU fails the production preflight.')
