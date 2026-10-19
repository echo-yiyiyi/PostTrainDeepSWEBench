"""Reuse PostTrain prompt generation and result artifacts in the Ibex launcher."""
import os
import shutil
import subprocess
import sys
from runtime import ROOT, TASK
from evaluation_config import TASK_ID, EXPERIMENT_MODE, TEACHING_NAME


def prepare_teaching_materials(task, mode=EXPERIMENT_MODE):
    if mode != 'task_specific_training':
        return
    destination = task / 'reference_materials'
    shutil.copytree(ROOT / 'src/deepswe/teaching_materials' / TEACHING_NAME / 'solution', destination / 'solution')
    shutil.copytree(TASK / 'tests', destination / 'tests')


def make_prompt(base, train_seconds, mode=EXPERIMENT_MODE):
    if mode != 'task_specific_training':
        return make_general_prompt(base, train_seconds)
    template = (ROOT / 'src/deepswe/prompts' / f'teach_{TEACHING_NAME}.txt').read_text()
    instruction = (TASK / 'instruction.md').read_text()
    if '<evaluation_task>\n' + instruction + '</evaluation_task>' not in template:
        raise ValueError('Teaching prompt must contain the complete, unchanged task instruction')
    # Replace only named placeholders, preserving literal JSON and TypeScript braces.
    return template.replace('{base_model}', base).replace('{train_minutes}', f'{train_seconds / 60:g}')


def make_general_prompt(base, train_seconds):
    full = os.environ.get('DEEPSWE_MODE') == 'full'
    prompt = subprocess.check_output([sys.executable,
        str(ROOT / 'src/eval/general/get_prompt.py'), '--agent', 'codex',
        '--model-to-train', base, '--benchmark-id', 'deepswe' if full else 'deepswe_smoke',
        '--num-hours', str(train_seconds / 3600)], cwd=ROOT, text=True)
    prompt = prompt.replace('The machine is equipped with an Nvidia H100 GPU.',
        'The GPU is allocated by Slurm; inspect nvidia-smi for its type and memory.')
    prompt = prompt.replace('The packages are installed via \\`uv pip install --system\\`.',
        'Use uv pip install --python /home/ben/.venv/bin/python PACKAGE in the writable virtual environment.')
    # The standard generator includes this section when the task has test_data.json.
    # DeepSWE derives the same checker reference from its vendored instruction.
    if full:
        return prompt + '''
Full-suite protocol: all 113 Datacurve DeepSWE tasks are held-out tests.
Do not download, inspect, reconstruct, or train on that dataset, its solutions,
tests, task-specific commits, or published agent traces. No test material is
mounted here. You may query aggregate benchmark scores via evaluate.py during
training, as in PostTrainBench; this never authorizes using test data for training.
Create independent software engineering data and validate your pipeline locally.
The designated starting checkpoint is Qwen/Qwen3.5-9B (already post-trained);
continue training that exact checkpoint, not another model. Its pinned revision
is recorded in model_profile.json, together with the predownloaded cached_model
path. Load from that path, or pass the exact revision to from_pretrained; do not
resolve a floating main revision. Use the supplied templates/qwen3_5.jinja
and mini_swe_bash_tool.json for text-only native tool conversations, with
enable_thinking=False. The final evaluator uses mini-swe-agent, 32768 total
context tokens and 4000 output tokens per call. Shell calls are independent.
Commit edits before echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT, issued alone.
Use AutoModelForImageTextToText/Qwen3_5ForConditionalGeneration as appropriate
for the supplied Transformers version; this is not a Qwen3 architecture.
Export complete Hugging Face weights, config, tokenizer and processor files
to final_model/. Merge adapters, retain the architecture and all required
components, and verify reload using the starting environment before finishing.
Environment and commands (same writable-venv and host-bridge setup as smoke):
- Work in /home/ben/task. Python is /home/ben/.venv/bin/python (also first on PATH).
- Install experimental packages with:
  uv pip install --python /home/ben/.venv/bin/python PACKAGE
- Check remaining time with bash timer.sh; inspect the allocated GPU with nvidia-smi.
- After exporting a COMPLETE checkpoint and freeing GPU memory, try:
  python evaluate.py --model-path final_model --templates-dir templates --limit 1 --json-output-file trial_metrics.json
- --limit N evaluates a deterministic prefix of N tasks; omit it or pass -1 for
  all 113. --max-tokens defaults to 4000 per model reply. Trials run sequentially,
  may be slow, count against your training budget, and are cancelled at the deadline.
- evaluate.py contacts the authenticated host bridge. Do not launch containers,
  install Docker, or start your own official evaluation server inside the sandbox.
- Task containers are separate from the training container: packages installed
  here are not available to the evaluated coding agent. Do not leave commands
  that depend on your training workspace in its generated task trajectories.
- Trial responses contain aggregate scores, not task questions, reference answers,
  tests or per-task traces. Full host diagnostics remain outside this sandbox.
Independent judges and the complete final evaluation execute after training.

'''
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
    env = dict(os.environ, EVAL_RUNTIME='deepswe_smoke', DEEPSWE_SOURCE_HOME=str(home))
    with (result / 'final_evaluation.log').open('w') as log:
        subprocess.run(['bash', 'src/eval/run_final_eval.sh', 'deepswe_smoke',
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
    source_config = json.loads((home.parent / 'run_config.json').read_text())
    metrics['harness'] = source_config.get('harness', 'legacy-json')
    if 'mini_revision' in source_config:
        metrics['mini_revision'] = source_config['mini_revision']
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
