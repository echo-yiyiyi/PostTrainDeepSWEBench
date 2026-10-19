"""Official mini-swe-agent rollout with Inspect scoring and an Ibex environment.

Only transport and request seeds are adapted. Agent prompts, tool parsing,
format-error handling, observations and submission detection are upstream.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import urllib.request

import litellm
import yaml
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, accuracy, scorer
from inspect_ai.solver import solver
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel

from evaluation_config import MINI_REVISION, TASK_DIRECTORY

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/eval'))
from per_sample_seed import sample_seed


def rpc(operation, body):
    req = urllib.request.Request(os.environ['DEEPSWE_WORKER_URL'] + operation,
        data=json.dumps(body).encode(), headers={
            'Authorization': 'Bearer ' + os.environ['DEEPSWE_WORKER_TOKEN'],
            'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=2100) as response:
        return json.load(response)


class TaskEnvironment(LocalEnvironment):
    def execute(self, action, cwd='', *, timeout=None):
        result = rpc('/mini-shell', dict(command=action['command'],
            cwd=cwd or self.config.cwd, env=self.config.env,
            timeout=timeout or self.config.timeout))
        output = {'output': result['output'], 'returncode': result['exit_code'],
                  'exception_info': result.get('exception_info', '')}
        self._check_finished(output)
        return output

    def get_template_vars(self, **kwargs):
        # Same system-information template, describing the TASK rather than host.
        return self.config.model_dump() | self.system_info | kwargs


class SeededModel(LitellmModel):
    def __init__(self, run_seed, task_id, **kwargs):
        super().__init__(**kwargs)
        self.run_seed, self.task_id, self.calls = run_seed, task_id, 0

    def _query(self, messages, **kwargs):
        kwargs['seed'] = sample_seed(self.run_seed, self.task_id, call=self.calls)
        self.calls += 1
        return super()._query(messages, **kwargs)


def run_agent(config, directory, seed, instruction):
    native = yaml.safe_load((Path(__file__).parent / 'mini_swe/mini.yaml').read_text())
    env = TaskEnvironment(cwd='/app', **native.get('environment', {}))
    result = rpc('/mini-shell', {'command':
        "python3 -c 'import platform,json; print(json.dumps(platform.uname()._asdict()))'"})
    if result['exit_code']:
        raise RuntimeError('Cannot read task system information')
    # Singularity may print cleanup diagnostics after command stdout as well.
    env.system_info = next(json.loads(line) for line in result['output'].splitlines()
                           if line.startswith('{"system":'))
    model_config = dict(native['model'])
    model_config.pop('model_class', None)
    model_config['model_name'] = 'openai/deepswe-qwen'
    model_config['cost_tracking'] = 'ignore_errors'
    model_config['model_kwargs'] = dict(model_config.get('model_kwargs', {}),
        api_base=os.environ['DEEPSWE_MODEL_URL'], api_key=os.environ['DEEPSWE_MODEL_KEY'],
        max_tokens=config['max_tokens'], timeout=120, num_retries=0)
    if config['temperature'] is not None:
        model_config['model_kwargs']['temperature'] = config['temperature']
    model = SeededModel(seed, config['task_id'], **model_config)
    agent_config = dict(native['agent'])
    # mini --yolo / --exit-immediately removes interactive confirmation; the
    # underlying DefaultAgent implements the same autonomous rollout.
    agent_config.pop('mode', None)
    agent_config.update(cost_limit=0, wall_time_limit_seconds=config['agent_timeout_seconds'],
                        output_path=directory / 'mini-swe-agent.trajectory.json')
    agent = DefaultAgent(model, env, **agent_config)
    try:
        result = agent.run(instruction)
    except litellm.exceptions.ContextWindowExceededError:
        # Native agent has already saved the exception and stopped. Grade its
        # committed work once; never feed this error back as an invalid action.
        result = {'exit_status': 'ContextWindowExceededError', 'submission': ''}
    finally:
        # Native trajectory serializes model_kwargs. Redact transient auth keys.
        model.config.model_kwargs.pop('api_key', None)
        agent.save(directory / 'mini-swe-agent.trajectory.json')
    (directory / 'agent_result.json').write_text(json.dumps(result, indent=2))
    return result


@solver
def mini_solver(config, directory, seed):
    async def solve(state, generate):
        result = await asyncio.to_thread(run_agent, config, directory, seed, state.input_text)
        state.output = ModelOutput.from_content('mini-swe-agent', result.get('submission', ''))
        state.metadata['agent_exit_status'] = result['exit_status']
        return state
    return solve


@scorer(metrics=[accuracy()])
def official_verifier(directory):
    async def score(state, target):
        reward = await asyncio.to_thread(rpc, '/verify', {})
        (directory / 'reward.json').write_text(json.dumps(reward, indent=2))
        return Score(value=reward['reward'], metadata=reward)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    config = json.loads((args.output / 'evaluation_config.json').read_text())
    task_path = Path(config.get('task_path', ROOT / 'src/eval/tasks/deepswe_smoke' / TASK_DIRECTORY))
    instruction = (task_path / 'instruction.md').read_text()
    task = Task(dataset=[Sample(id=config['task_id'], input=instruction)],
        solver=mini_solver(config, args.output, args.seed), scorer=official_verifier(args.output))
    # Inspect owns scoring/log status; generation is entirely the native agent.
    logs = inspect_eval(task, model='mockllm/orchestration-only',
        log_dir=str(args.output / 'inspect_logs'), log_format='json', display='none')
    if len(logs) != 1 or logs[0].status != 'success':
        raise RuntimeError('Native mini-swe evaluation failed; no score is produced')
    reward = json.loads((args.output / 'reward.json').read_text())
    result = json.loads((args.output / 'agent_result.json').read_text())
    metrics = dict(accuracy=reward['reward'], num_tasks=1, seed=args.seed,
        reward=reward, single_task_test=not config.get('held_out_evaluation', False), task_id=config['task_id'],
        experiment_mode=config['experiment_mode'], harness='mini-swe-agent',
        mini_revision=MINI_REVISION, agent_exit_status=result['exit_status'],
        summarization=False, training_task_exposed=not config.get('held_out_evaluation', False),
        reference_solution_exposed=not config.get('held_out_evaluation', False),
        verifier_tests_exposed=not config.get('held_out_evaluation', False),
        held_out_evaluation=config.get('held_out_evaluation', False))
    if config.get('infrastructure_smoke'):
        metrics.update(infrastructure_smoke=True, experiment_mode='infrastructure_smoke',
                       training_task_exposed=None, reference_solution_exposed=None,
                       verifier_tests_exposed=None)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
