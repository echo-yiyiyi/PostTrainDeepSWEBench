"""Verify all allocated GPUs through the actual training container launcher."""
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cluster/ibex/deepswe'))
from run import TRAIN_ENV, prepare_training_home
from runtime import container, run

expected = int(os.environ['SLURM_GPUS_ON_NODE'])
assert expected == 2, expected
image = Path(os.environ['DEEPSWE_STORAGE_IMAGES']) / 'qwen35_v032.sif'
report = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / ('multi_gpu_check_' + os.environ['SLURM_JOB_ID'])
report.mkdir(exist_ok=False)
with tempfile.TemporaryDirectory(dir=os.environ['DEEPSWE_CONTAINER_TMPDIR']) as directory:
    home = Path(directory) / 'home'
    home.mkdir()
    prepare_training_home(home, modern=True)
    with (report / 'probe.log').open('w') as log:
        run(container(image, home, gpu=True, env={
            **TRAIN_ENV, 'DEEPSWE_REQUIRE_CUDA': '1', 'DEEPSWE_EXPECT_GPUS': str(expected)}) +
            ['bash', '/home/ben/setup_training_env.sh'], timeout=300, stdout=log)
        code = (
            'import json,os,torch; '
            f'assert torch.cuda.device_count() == {expected}; '
            'values=[torch.ones(1,device=f"cuda:{i}").item() for i in range(torch.cuda.device_count())]; '
            'print(json.dumps(dict(passed=True,visible=os.environ["CUDA_VISIBLE_DEVICES"],'
            'devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],values=values)))'
        )
        _, output = run(container(image, home, gpu=True, env=TRAIN_ENV) +
                        ['python', '-c', code], timeout=300)
        log.write(output)
        print(output)
        (report / 'ready.json').write_text(next(line for line in output.splitlines()
                                              if line.startswith('{"passed":')) + '\n')
