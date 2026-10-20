"""Check all dependencies, recognizing the official vLLM image's NCCL override.

vLLM 0.32.0 intentionally overrides torch's NCCL pin for DeepEPv2; see its
pinned docker/Dockerfile. No other mismatch is permitted.
"""
import importlib.metadata
from pathlib import Path
import subprocess


def check() -> None:
    result = subprocess.run(['uv', 'pip', 'check'], capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, end='')
    if result.returncode == 0:
        return
    expected = ('The package `torch` requires `nvidia-nccl-cu12==2.29.7 ; '
                "sys_platform == 'linux'`, but `2.30.7` is installed")
    diagnostics = [line for line in output.splitlines() if line.startswith('The package ')]
    override = Path('/etc/uv-overrides.txt')
    if (result.returncode == 1 and diagnostics == [expected]
            and importlib.metadata.version('vllm') in ('0.32.0', '0.32.0+cu129')
            and importlib.metadata.version('torch').split('+')[0] == '2.13.0'
            and importlib.metadata.version('nvidia-nccl-cu12') == '2.30.7'
            and override.is_file()
            and 'nvidia-nccl-cu12==2.30.7' in override.read_text().splitlines()):
        print('Accepted ONLY the official vLLM 0.32.0 NCCL 2.30.7 override for DeepEPv2.')
        return
    raise RuntimeError('Unexpected dependency mismatch; refusing image activation')


if __name__ == '__main__':
    check()
