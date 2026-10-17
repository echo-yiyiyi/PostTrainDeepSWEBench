"""Trusted single-task Inspect/vLLM evaluation; shell operations run on the host worker."""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import solver, system_message

sys.path.insert(0, '/')
from harness_seed import per_sample_seed

PROTOCOL = '''You are solving a software engineering task in /app.
Respond with exactly one JSON object per turn: {"command": "a bash command"}.
Commands execute in /app in the task environment; edits to /app persist.
Inspect and edit the repository and test your changes. Before finishing, commit
all your changes with git (configure user.name/user.email locally if needed).
Only committed changes are graded. When finished respond {"done": true}.
Do not print markdown fences. You have at most 12 command turns.'''


def rpc(operation, body):
    req = urllib.request.Request(os.environ['DEEPSWE_WORKER_URL'] + operation,
        data=json.dumps(body).encode(), headers={
            'Authorization': 'Bearer ' + os.environ['DEEPSWE_WORKER_TOKEN'],
            'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


@solver
def shell_loop():
    async def solve(state, generate):
        for _ in range(12):
            # Keep the instruction plus recent command/observation pairs within the
            # small smoke-test context budget. Full events remain in Inspect logs.
            while len(state.messages) > 4 and sum(len(str(m.content)) for m in state.messages) > 12000:
                del state.messages[2:4]
            state = await generate(state)
            try:
                action = json.loads(state.output.completion)
                if action.get('done') is True:
                    break
                if not isinstance(action.get('command'), str):
                    raise ValueError('Expected command string')
            except (ValueError, AttributeError):
                observation = 'Invalid action. Return exactly {"command":"..."} or {"done":true}.'
            else:
                observation = json.dumps(rpc('/shell', {'command': action['command']}))
            state.messages.append(ChatMessageUser(content=observation))
        return state
    return solve


@scorer(metrics=[accuracy()])
def verifier():
    async def score(state, target):
        result = rpc('/verify', {})
        Path('/output/reward.json').write_text(json.dumps(result, indent=2))
        return Score(value=result['reward'], answer=state.output.completion, metadata=result)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    task = Task(dataset=[Sample(id='fastapi-implicit-head-options',
        input=Path('/instruction.md').read_text())],
        solver=[system_message(PROTOCOL), per_sample_seed(shell_loop(), args.seed)], scorer=verifier())
    logs = inspect_eval(task, model='vllm//model', model_args={
        'chat_template': '/templates/qwen3.jinja', 'gpu_memory_utilization': 0.75,
        'max_model_len': 8192}, log_dir='/output/inspect_logs', log_format='json',
        max_tokens=768, temperature=0.0, max_connections=1, score_display=False)
    if len(logs) != 1 or logs[0].status != 'success':
        raise RuntimeError('Inspect evaluation failed; no score is produced')
    reward = json.loads(Path('/output/reward.json').read_text())
    Path('/output/metrics.json').write_text(json.dumps({
        'accuracy': reward['reward'], 'num_tasks': 1, 'seed': args.seed,
        'smoke_only': True, 'reward': reward,
        'task_id': 'fastapi-implicit-head-options'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
