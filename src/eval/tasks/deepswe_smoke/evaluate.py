"""Agent-facing client: request evaluation from the host, no nested containers."""
import argparse
import json
import os
import urllib.request
import urllib.error
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default='final_model')
    p.add_argument('--json-output-file', default='metrics.json')
    p.add_argument('--templates-dir')  # Common benchmark CLI compatibility.
    p.add_argument('--max-tokens', type=int, default=4000)
    p.add_argument('--limit', type=int, default=-1)
    p.add_argument('--seed', type=int, default=72332)
    a = p.parse_args()
    if a.limit not in (-1, 1):
        p.error('This benchmark contains one task; --limit must be 1 or -1')
    body = json.dumps({'model_path': str(Path(a.model_path).resolve()), 'seed': a.seed, 'max_tokens': a.max_tokens}).encode()
    req = urllib.request.Request(os.environ['DEEPSWE_EVAL_URL'] + '/evaluate', data=body,
        headers={'Authorization': 'Bearer ' + os.environ['DEEPSWE_EVAL_TOKEN'], 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=14400) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        try:
            detail = json.loads(detail).get('error', detail)
        except ValueError:
            pass
        raise RuntimeError(f'Evaluation HTTP {error.code}: {detail}') from error
    if 'error' in result:
        raise RuntimeError(result['error'])
    Path(a.json_output_file).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
