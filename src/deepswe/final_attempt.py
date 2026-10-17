"""One isolated attempt, called by PostTrain's shared final-evaluation cascade."""
import argparse
import json
import os
from pathlib import Path
from runtime import Bridge, resolve_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--json-output-file', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--max-tokens', type=int, default=4000)
    args = parser.parse_args()
    home = Path(os.environ['DEEPSWE_SOURCE_HOME']).resolve()
    checkpoint = resolve_checkpoint('/home/ben/task/final_model', home)
    if checkpoint != args.model_path.resolve():
        raise ValueError('Final evaluation must use the submitted checkpoint')
    images = {name: Path(os.environ[f'DEEPSWE_{name.upper()}_SIF']).resolve()
              for name in ('eval', 'task')}
    bridge = Bridge(home, args.json_output_file.parent, images)
    bridge.accept_agent = False
    bridge.start()
    try:
        metrics = bridge.evaluate('/home/ben/task/final_model', args.seed, args.max_tokens)
        args.json_output_file.write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2))
    finally:
        bridge.server.shutdown()
        bridge.server.server_close()


if __name__ == '__main__':
    main()
