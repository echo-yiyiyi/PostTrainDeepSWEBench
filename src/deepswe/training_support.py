"""Reuse PostTrain prompt generation and result artifacts in the Ibex launcher."""
import os
import shutil
import subprocess
import sys
from runtime import ROOT, TASK
from evaluation_config import TASK_ID, EXPERIMENT_MODE


def prepare_teaching_materials(task, mode=EXPERIMENT_MODE):
    if mode != 'task_specific_training':
        return
    destination = task / 'reference_materials'
    shutil.copytree(ROOT / 'src/deepswe/teaching_materials/true_myth/solution', destination / 'solution')
    shutil.copytree(TASK / 'tests', destination / 'tests')


def make_prompt(base, train_seconds, mode=EXPERIMENT_MODE):
    if mode != 'task_specific_training':
        return make_general_prompt(base, train_seconds)
    template = (ROOT / 'src/deepswe/prompts/teach_true_myth.txt').read_text()
    instruction = (TASK / 'instruction.md').read_text()
    if '<evaluation_task>\n' + instruction + '</evaluation_task>' not in template:
        raise ValueError('Teaching prompt must contain the complete, unchanged task instruction')
    # Replace only named placeholders, preserving literal JSON and TypeScript braces.
    return template.replace('{base_model}', base).replace('{train_minutes}', f'{train_seconds / 60:g}')


def make_general_prompt(base, train_seconds):
    prompt = subprocess.check_output([sys.executable,
        str(ROOT / 'src/eval/general/get_prompt.py'), '--agent', 'codex',
        '--model-to-train', base, '--benchmark-id', 'deepswe',
        '--num-hours', str(train_seconds / 3600)], cwd=ROOT, text=True)
    prompt = prompt.replace('The machine is equipped with an Nvidia H100 GPU.',
        'The GPU is allocated by Slurm; inspect nvidia-smi for its type and memory.')
    prompt = prompt.replace('The packages are installed via \\`uv pip install --system\\`.',
        'Use uv pip install --python /home/ben/.venv/bin/python PACKAGE in the writable virtual environment.')
    # The standard generator includes this section when the task has test_data.json.
    # DeepSWE derives the same checker reference from its vendored instruction.
    if '## Decontamination Tool' not in prompt:
        prompt += ('\nThe held-out test_data.json at ../test_data.json is ONLY input to '
            '../contamination_check.py, not training data. Reading and running these two files '
            'is an allowed exception to the working-directory rule. Check independent data with '
            'python ../contamination_check.py --reference ../test_data.json --input train_data.jsonl.\n')
    return prompt + '''
DeepSWE interface: this run evaluates one held-out repository task. You train
Qwen; the trained Qwen executes the task. See evaluation_protocol.txt and
evaluation_config.json for the exact protocol and test limits. The fixed
chat template is templates/qwen3.jinja. Save full Hugging Face weights and
tokenizer in final_model (merge LoRA adapters if used).
The task container is separate from this training container; installed tools
and Python packages can differ. evaluate.py contacts the host evaluator; do
not launch containers. Free GPU memory before evaluating. Exploratory
evaluations consume your training budget. The standard independent judges
run after training and before final evaluation.
'''


def save_artifacts(result, home, elapsed):
    seconds = int(elapsed)
    (result / 'time_taken.txt').write_text(f'{seconds//3600:02}:{seconds//60%60:02}:{seconds%60:02}\n')
    subprocess.run([sys.executable, str(ROOT / 'src/trace_parsing/parse_trace.py'),
        '--agent', 'codex', str(result / 'solve_out.txt'), '-o', str(result / 'solve_parsed.txt')],
        cwd=ROOT, check=True)
    for name in ['cli_version.txt']:
        for source in [home / 'task' / name, home / name]:
            if source.exists():
                shutil.copy2(source, result / name)
                break
    if not (result / 'task').exists():
        (result / 'task').symlink_to(home / 'task', target_is_directory=True)


def final_evaluation(result, home):
    env = dict(os.environ, EVAL_RUNTIME='deepswe', DEEPSWE_SOURCE_HOME=str(home))
    with (result / 'final_evaluation.log').open('w') as log:
        subprocess.run(['bash', 'src/eval/run_final_eval.sh', 'deepswe',
            str(home / 'task/final_model'), str(result / 'final_eval'),
            str(result / 'metrics.json')], cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, check=True)
    if not (result / 'metrics.json').is_file():
        raise RuntimeError('Final evaluation exhausted all shared retries; see final_evaluation.log')
    import json
    metrics_path = result / 'metrics.json'
    metrics = json.loads(metrics_path.read_text())
    metrics.update(task_id=TASK_ID, experiment_mode=EXPERIMENT_MODE,
                   training_task_exposed=True, reference_solution_exposed=True, verifier_tests_exposed=True,
                   held_out_evaluation=False)
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
