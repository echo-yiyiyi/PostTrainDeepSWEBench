"""CPU-only prompt and network probe inside the training image."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cluster/ibex/deepswe"))
from run import TRAIN_ENV, prepare_training_home, prepare_protocol_files
from runtime import cache_overlay, container, run
from model_profiles import QWEN35_ID

image = Path(os.environ["DEEPSWE_STORAGE_IMAGES"]) / "qwen35_v032.sif"
report = Path(os.environ["DEEPSWE_STORAGE_RESULTS"]) / (
    "prompt_network_" + os.environ["SLURM_JOB_ID"])
report.mkdir(exist_ok=False)
with tempfile.TemporaryDirectory(dir=os.environ["DEEPSWE_CONTAINER_TMPDIR"]) as directory:
    home = Path(directory) / "home"
    (home / "task").mkdir(parents=True)
    (home / "tmp").mkdir()
    (home / ".cache/uv").mkdir(parents=True)
    probe_env = {**TRAIN_ENV, "TMPDIR": "/home/ben/tmp",
                 "XDG_CACHE_HOME": "/home/ben/.cache",
                 "UV_CACHE_DIR": "/home/ben/.cache/uv"}
    prepare_training_home(home, modern=True)
    prepare_protocol_files(home, QWEN35_ID)
    shutil.copy2(ROOT / "src/eval/tasks/deepswe/evaluate.py", home / "task/evaluate.py")
    (home / "task/timer.sh").write_text(
        '#!/bin/bash\n'
        f'remaining=$(({int(time.time()) + 900} - $(date +%s)))\n'
        'printf "Remaining time (minutes:seconds): %d:%02d\\n" '
        '"$((remaining / 60))" "$((remaining % 60))"\n')
    with (report / "probe.log").open("w") as log, cache_overlay(
            Path(os.environ["HF_HOME"]), Path(directory) / "cache") as cache:
        try:
            run(container(image, home, [
                f'{cache}:/home/ben/hf_cache',
                f'{ROOT / "dev_utils/deepswe/prompt_network_probe.py"}:/network_probe.py:ro',
                f'{ROOT / "dev_utils/deepswe/prompt_environment_probe.py"}:/audit.py:ro'],
                network=True, gpu=False, pwd="/home/ben/task", env=probe_env) +
                ["bash", "-ec",
                 "uv venv --python /usr/bin/python --system-site-packages /home/ben/.venv\n"
                 "python /audit.py --cpu-only\n"
                 "python /network_probe.py"], timeout=600, stdout=log)
        finally:
            for name in ("prompt_network.json", "prompt_environment.json"):
                result = home / "task" / name
                if result.exists():
                    (report / name).write_bytes(result.read_bytes())
print(f"Prompt files, commands, network and online package installation passed: {report}")
