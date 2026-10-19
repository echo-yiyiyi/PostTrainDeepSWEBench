# DeepSWE full benchmark

This directory follows the task layout used by the other benchmarks:
`benchmark.txt`, `info.json`, `evaluate.py`, and `evaluate_final_eval.py`.
`prompt.txt` holds the task-specific supplement to `src/eval/general/prompt.txt`;
`prompt.preview.txt` is the rendered 10-hour Qwen3.5 prompt, including Ibex setup.
`api_policy.py` is shared by the prompt generator and API judge.
`evaluation_config.py`, `model_profiles.py`, and `mini_swe/` define the evaluation
settings, model-specific serving protocol and pinned upstream agent assets.

The other Python modules here are **host-only** evaluation implementation, not
agent sandbox assets. They are deliberately not in `evaluation_code/`, which
the generic launcher copies into the training sandbox. Only `evaluate.py` and
explicitly selected public protocol files are copied by the Ibex launcher.
Test data stays in the external host manifest/cache; it is never vendored here.

Cluster setup and orchestration live in [cluster/ibex/deepswe](../../../../cluster/ibex/deepswe/README.md).
Submission entry points remain in `src/commit_utils/`. Smoke-specific task
selection and teaching assets live in [deepswe_smoke](../deepswe_smoke/README.md).

## Full held-out suite with Qwen3.5-9B

The `deepswe` mode is separate from the `deepswe_smoke` single-task teaching experiment.
All **113 Datacurve DeepSWE tasks** at revision
`0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea` are tests. There is no train/test split
of this dataset. Codex designs and synthesizes independent training data.
Test prompts, repositories, reference solutions and verifiers are not mounted in
the training container. Like PostTrainBench, `evaluate.py` supports training-time
checkpoint evaluation through a host bridge, returning aggregate scores only.
Use `--limit 1` for a small trial or omit the limit to evaluate all 113 tasks.
Trials consume the training budget and are cancelled at its deadline. Published DeepSWE material
and task-derived synthetic data are prohibited by both prompt and judge policy.
Independent synthetic development examples remain available for local validation.

The designated initial checkpoint is **Qwen/Qwen3.5-9B**, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` (the officially post-trained model,
not a fictional Qwen3.5-9B-Base). We retain the complete multimodal architecture
while training/evaluating on text. Its native pinned chat template is
`src/eval/templates/qwen3_5.jinja`, with `qwen3_coder` tool parsing and thinking
disabled. Do not substitute the Qwen3 JSON-tool template. The dedicated image
uses vLLM 0.17.1 and Transformers 5.2.0; the older images are untouched.
Final evaluation uses 32768 context tokens and 4000 output tokens per call,
overriding only the checkpoint's output-length default.

The prompt is generated from the normal PostTrainBench template. Full-mode
changes are: aggregate-only training trial feedback, the designated Qwen3.5 checkpoint,
actual Slurm GPU information, and the explicitly requested external-API policy.
**Codex may serve as the synthetic-data pipeline agent**: design, implement,
execute and debug the pipeline, and author examples in its own agent work.
Small auxiliary API calls for pipeline checks and non-generative data processing
are allowed and must be logged in `external_api_usage.jsonl`. Bulk hosted teacher
completions (answers, patches, reasoning chains, trajectories, labels or rewards)
for distillation are prohibited. Both training prompt and API judge import the
same `api_policy.py`. Ordinary benchmark API policies remain unchanged.

Preparation (reads paths and credentials from this checkout's `.env`):

```bash
sbatch --account=conf-cvpr-2026.11.23-ghanembs --gres=gpu:a100:1 \
  --cpus-per-task=8 --mem=96G --time=02:00:00 \
  --output=logs/deepswe_qwen35_prepare_%j.out \
  --error=logs/deepswe_qwen35_prepare_%j.err cluster/ibex/deepswe/prepare_full_job.sh
```

This caches all model shards and processor/tokenizer assets under configured
`HF_HOME`, records the 113 host-only tasks and their file hashes, builds a candidate
image, and gates activation on a real 9B LoRA update/export and native vLLM tool
call. `qwen35_ready.json` is written only after these checks pass. Existing active
images are never overwritten. Task SIFs are pulled on demand into node-local
storage by trial and final-evaluation jobs; caching 113 large images is not required.

After preparation succeeds:

```bash
bash src/commit_utils/commit_deepswe.sh --dry-run
bash src/commit_utils/commit_deepswe.sh
```

The launcher submits a 10-hour Codex training job (12-hour allocation, including
judges), then a dependent 113-element final-evaluation array, then complete-only
aggregation. Each job uses the conference account and a GPU; array concurrency
defaults to 4 (`DEEPSWE_FULL_EVAL_CONCURRENCY`). `DEEPSWE_FULL_TRAIN_SECONDS` and
`DEEPSWE_FULL_WALLTIME` configure the training budget/allocation. Final-evaluation jobs cannot
start before training and judges finish successfully; optional training trials
run within the training allocation. Independent task containers
and pristine verifier workspaces retain upstream tests unchanged. Inference and
test containers receive no third-party provider credentials.

Results: `$DEEPSWE_STORAGE_RESULTS/deepswe/TRAIN_JOB_ID/`. All test outputs
are under `final_eval/TASK_ID/` (training trials under `trial_*/TASK_ID/`); trajectories, submitted patches and test logs are
retained, while copied source/dependency trees are removed after scoring. No
full-suite `metrics.json` is written if any task has an infrastructure failure or
is missing; zeros from completed model attempts count normally. The aggregate
is raw model accuracy; independent judge verdicts remain alongside it and must
be reviewed before presenting an integrity-qualified score. No baseline-adjusted
PostTrainBench leaderboard score is claimed for this new task.

Validation: `python3 -m unittest discover -s dev_utils/deepswe -p test_full.py`
checks policy alignment, training-evaluation instructions, task integrity,
complete-only aggregation and model-specific protocol selection. These CPU checks
do not replace the required GPU image gate.

The full prompt uses the same writable `/home/ben/.venv` and host evaluation
bridge command interface as smoke. Install packages with
`uv pip install --python /home/ben/.venv/bin/python PACKAGE`, and release training
GPU allocations before invoking `python evaluate.py`. Full mode preserves the
new image’s CUDA/NCCL environment instead of injecting smoke’s old CUDA paths.
The Qwen3.5 template and modern image remain model-specific requirements.
`dev_utils/deepswe/test_trial_bridge.py` checks the client/bridge interface,
aggregate-only feedback, deadline cancellation and shell environment.
