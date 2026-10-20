"""Host orchestrator: compare a rebuilt training SIF with the unchanged eval SIF."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval/tasks/deepswe'))
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from runtime import container, run
from run import prepare_training_home, TRAIN_ENV

candidate = Path(sys.argv[1]).resolve()
output = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('image_compat_' + os.environ['SLURM_JOB_ID'])
output.mkdir()
scratch = Path(os.environ['DEEPSWE_CONTAINER_TMPDIR']) / 'compat'
checkpoint = scratch / 'checkpoint'
checkpoint.mkdir(parents=True)
cache = Path(os.environ['HF_HOME']) / 'hub'
snapshots = cache / 'models--Qwen--Qwen3-1.7B-Base/snapshots'
base = sorted(snapshots.iterdir())[-1]
# Bind the whole cache: HF snapshots contain relative symlinks into blobs.
base_in_container = '/cache/' + str(base.relative_to(cache))
for phase, image in [('train', candidate), ('eval', Path(os.environ['DEEPSWE_EVAL_SIF']))]:
    home = scratch / (phase + '_home')
    home.mkdir()
    checkpoint_bind = f'{checkpoint}:/checkpoint' + (':ro' if phase == 'eval' else '')
    binds = [f'{output}:/output', f'{cache}:/cache:ro', checkpoint_bind,
        f'{ROOT / "src/eval/templates"}:/templates:ro',
        f'{ROOT / "dev_utils/deepswe/image_compat_probe.py"}:/probe.py:ro']
    if phase == 'train':
        prepare_training_home(home)
    command = ['bash', '-c',
        ('bash /home/ben/setup_training_env.sh && ' if phase == 'train' else '') +
        f'python /probe.py {phase}']
    if phase == 'eval':
        binds += [f'{ROOT / "src/eval/tasks/deepswe"}:/harness:ro',
            f'{ROOT / "src/eval/per_sample_seed.py"}:/harness_seed.py:ro',
            f'{ROOT / "dev_utils/deepswe/test_inspect.py"}:/inspect_probe.py:ro']
        command[-1] += ' && python /inspect_probe.py'
    env = {'PYTHONNOUSERSITE': '1', 'COMPAT_BASE': base_in_container}
    if phase == 'train':
        env.update(TRAIN_ENV)
        env['DEEPSWE_REQUIRE_CUDA'] = '1'
    with (output / f'{phase}.log').open('w') as log:
        run(container(image, home, binds, gpu=True, network=True, env=env) + command,
            timeout=900, stdout=log)
    print('PASS:', phase, flush=True)
train = json.loads((output / 'train_packages.json').read_text())
evaluation = json.loads((output / 'eval_packages.json').read_text())
differences = {k: {'train': train.get(k), 'eval': evaluation.get(k)}
    for k in sorted(train.keys() | evaluation.keys()) if train.get(k) != evaluation.get(k)}
(output / 'package_differences.json').write_text(json.dumps(differences, indent=2))
pins = dict(line.strip().split('==', 1) for line in
    (ROOT / 'containers/requirements-direct.txt').read_text().splitlines()
    if line.strip() and not line.startswith('#'))
pin_drift = {phase: {name: {'expected': version, 'actual': packages.get(name)}
    for name, version in pins.items() if packages.get(name) != version}
    for phase, packages in [('train', train), ('eval', evaluation)]}
(output / 'direct_dependency_drift.json').write_text(json.dumps(pin_drift, indent=2))
core = ['torch', 'transformers', 'tokenizers', 'safetensors', 'vllm', 'peft',
        'accelerate', 'huggingface-hub']
assert not any(k in differences for k in core), differences
assert not pin_drift['train'], pin_drift
(output / 'PASS').write_text(str(candidate) + '\n')
print('PASS: core dependencies match; training export loads and generates in fixed evaluation image')
