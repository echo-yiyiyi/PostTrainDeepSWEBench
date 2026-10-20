"""Exercise fakeroot + --nv in both ML SIFs on an allocated GPU, without an LLM API."""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval/tasks/deepswe'))
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from runtime import container, run
from run import prepare_training_home, TRAIN_ENV

probe = '''import os, torch
assert os.getuid() == 0
assert torch.cuda.is_available()
x = torch.arange(4, device='cuda', dtype=torch.float32)
assert x.sum().item() == 6
print('PASS: UID 0 and CUDA tensor operation:', torch.cuda.get_device_name(0))
'''
assert os.getuid() != 0, 'Run as the ordinary Slurm user'
with tempfile.TemporaryDirectory(prefix='gpu-fakeroot-', dir=ROOT / 'results') as tmp:
    root = Path(tmp)
    for name in ('train', 'eval'):
        home = root / name
        home.mkdir()
        image = Path(os.environ[f'DEEPSWE_{name.upper()}_SIF'])
        (home / 'cuda_probe.py').write_text(probe)
        if name == 'train':
            prepare_training_home(home)
            cmd = container(image, home, gpu=True, network=True,
                            env={**TRAIN_ENV, 'DEEPSWE_REQUIRE_CUDA': '1'})
            command = ['bash', '-c', 'set -e; bash /home/ben/setup_training_env.sh; python /home/ben/cuda_probe.py']
        else:
            cmd = container(image, home, gpu=True, network=True, env={'PYTHONNOUSERSITE': '1'})
            command = ['python', '/home/ben/cuda_probe.py']
        print(f'Checking {name} image', flush=True)
        _, output = run(cmd + command, timeout=240)
        print(output, flush=True)
