"""Trusted single-task Inspect/vLLM evaluation; shell operations run on the host worker."""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ChatMessageSystem
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import solver

sys.path.insert(0, '/')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_seed import per_sample_seed

from evaluation_config import PROTOCOL


@solver
def protocol_message():
    async def solve(state, generate):
        # system_message formats braces; the action protocol must be literal JSON.
        state.messages.insert(0, ChatMessageSystem(content=PROTOCOL))
        return state
    return solve


def rpc(operation, body):
    req = urllib.request.Request(os.environ['DEEPSWE_WORKER_URL'] + operation,
        data=json.dumps(body).encode(), headers={
            'Authorization': 'Bearer ' + os.environ['DEEPSWE_WORKER_TOKEN'],
            'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=2100) as response:
        return json.load(response)


@solver
def shell_loop(agent_timeout=10800):
    async def solve(state, generate):
        deadline = time.monotonic() + agent_timeout
        while time.monotonic() < deadline:
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
    config = json.loads(Path('/output/evaluation_config.json').read_text())
    task = Task(dataset=[Sample(id=config['task_id'],
        input=Path('/instruction.md').read_text())],
        solver=[protocol_message(), per_sample_seed(shell_loop(config["agent_timeout_seconds"]), args.seed)], scorer=verifier())
    logs = inspect_eval(task, model='vllm//model', model_args={
        'chat_template': '/templates/qwen3.jinja', 'gpu_memory_utilization': config['gpu_memory_utilization']}, log_dir='/output/inspect_logs', log_format='json',
        max_tokens=config['max_tokens'], max_connections=1, score_display=False)
    if len(logs) != 1 or logs[0].status != 'success':
        raise RuntimeError('Inspect evaluation failed; no score is produced')
    reward = json.loads(Path('/output/reward.json').read_text())
    Path('/output/metrics.json').write_text(json.dumps({
        'accuracy': reward['reward'], 'num_tasks': 1, 'seed': args.seed,
        'single_task_test': True, 'reward': reward,
        'task_id': config['task_id'], 'experiment_mode': config['experiment_mode'],
        'training_task_exposed': True, 'reference_solution_exposed': True,
        'verifier_tests_exposed': True, 'held_out_evaluation': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
