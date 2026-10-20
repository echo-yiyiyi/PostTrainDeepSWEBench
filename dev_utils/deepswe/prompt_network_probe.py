"""Check external access and a real uncached install inside the training sandbox."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    results = {}
    for name, url in {
        "pypi": "https://pypi.org/simple/idna/",
        "huggingface": "https://huggingface.co/api/models/Qwen/Qwen3.5-9B",
        "github": "https://github.com",
    }.items():
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                response.read(1024)
                results[name] = {"passed": True, "status": response.status}
        except Exception as error:
            results[name] = {"passed": False, "error": str(error)}
    commands = {
        "git_https": ["git", "ls-remote", "https://github.com/pypa/sampleproject.git", "HEAD"],
        "uv_online_install": [
            "uv", "--no-cache", "pip", "install", "--python", sys.executable,
            "--index-url", "https://pypi.org/simple", "--no-deps",
            "--reinstall", "idna==3.10",
        ],
        "installed_import": [sys.executable, "-c", "import idna; assert idna.__version__ == '3.10'"],
    }
    for name, command in commands.items():
        try:
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            results[name] = {"passed": result.returncode == 0,
                             "output": (result.stdout + result.stderr)[-4000:]}
        except Exception as error:
            results[name] = {"passed": False, "error": str(error)}
    report = {"passed": all(item["passed"] for item in results.values()), "checks": results}
    Path("/home/ben/task/prompt_network.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
