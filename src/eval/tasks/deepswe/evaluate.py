"""Agent-facing client: request evaluation from the host, no nested containers."""
import argparse
import json
import os
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default='final_model')
    p.add_argument('--json-output-file', default='metrics.json')
    p.add_argument('--templates-dir')  # Common benchmark CLI compatibility.
    p.add_argument('--seed', type=int, default=72332)
    a = p.parse_args()
    body = json.dumps({'model_path': str(Path(a.model_path).resolve()), 'seed': a.seed}).encode()
    req = urllib.request.Request(os.environ['DEEPSWE_EVAL_URL'] + '/evaluate', data=body,
        headers={'Authorization': 'Bearer ' + os.environ['DEEPSWE_EVAL_TOKEN'], 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        result = json.load(response)
    if 'error' in result:
        raise RuntimeError(result['error'])
    Path(a.json_output_file).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
