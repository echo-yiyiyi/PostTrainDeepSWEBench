"""Regression coverage for the actual CUDA-image NCCL exception."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "dependency_gate", ROOT / "containers/patches/check_qwen35_dependencies.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

NCCL = ("The package `torch` requires `nvidia-nccl-cu12==2.29.7 ; "
        "sys_platform == 'linux'`, but `2.30.7` is installed\n")


class DependencyGateTest(unittest.TestCase):
    def run_gate(self, version, output=NCCL, override=True):
        versions = {"vllm": version, "torch": "2.13.0+cu129",
                    "nvidia-nccl-cu12": "2.30.7"}
        with (
            patch.object(gate.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 1, stdout=output, stderr="")),
            patch.object(gate.importlib.metadata, "version", side_effect=versions.__getitem__),
            patch.object(gate.Path, "is_file", return_value=override),
            patch.object(gate.Path, "read_text", return_value="nvidia-nccl-cu12==2.30.7\n"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            gate.check()

    def test_official_cuda_build_is_accepted(self):
        for version in ("0.32.0", "0.32.0+cu129"):
            with self.subTest(version=version):
                self.run_gate(version)

    def test_other_builds_are_rejected(self):
        for version in ("0.32.0+cu130", "0.33.0", "0.32.0rc1"):
            with self.subTest(version=version), self.assertRaises(RuntimeError):
                self.run_gate(version)

    def test_additional_conflict_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_gate("0.32.0+cu129", NCCL + "The package `example` requires `other`\n")

    def test_override_evidence_is_required(self):
        with self.assertRaises(RuntimeError):
            self.run_gate("0.32.0+cu129", override=False)


if __name__ == "__main__":
    unittest.main()
