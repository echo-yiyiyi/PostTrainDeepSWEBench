# DeepSWE post-training on Ibex

Canonical names: **deepswe_smoke** is the single-task teaching/debug mode;
**deepswe** is the complete 113-task held-out benchmark. The corresponding
submission scripts are `commit_deepswe_smoke.sh` and `commit_deepswe.sh`.
See “Full held-out suite with Qwen3.5-9B” below for the formal run.

## deepswe_smoke

Each evaluation attempt configures a repository-local Git identity in its fresh
`agent_app` (`DeepSWE Agent <agent@deepswe.local>`), before model execution.
Setup failures abort the attempt and diagnostics go to `git_setup.log`. The model
still creates its own commits; the verifier still collects only `BASE..HEAD`.
This avoids missing author metadata preventing otherwise valid commit commands,
without changing global Git configuration or editing previous evaluation results.

One **outer Codex `gpt-6-sol` training run with reasoning effort `low`**, fine-tuning
`Qwen/Qwen3-1.7B-Base`, followed by one evaluation of that checkpoint on the
`happy-dom-abort-pending-body-reads` DeepSWE task. Codex is the training agent; the
trained Qwen model is the agent operating on the evaluation repository.
This is not a direct evaluation of Codex's repository-solving ability.

## Active special teaching experiment

This run intentionally teaches the exact Happy DOM task “Abort pending body reads on shutdown”. It is **not a held-out
benchmark evaluation**. The complete task question is included verbatim in
`prompts/teach_happy_dom.txt`; `prompts/teach_happy_dom.rendered.txt` is the actual
60-minute prompt for Qwen/Qwen3-1.7B-Base.

For this special prompt only, the user authorizes use of the official reference
solution and tests. At training startup they are copied to
`/home/ben/task/reference_materials/{solution,tests}`. Codex may use those files to
build task-specific training data. These helper files are not mounted into the
small model's task container during final evaluation: it must produce the edits
from its trained weights. Normal prompting and its data rules remain available;
`prepare_teaching_materials(..., mode='general')` copies no reference materials.

Results record `experiment_mode=task_specific_training`,
`training_task_exposed=true`, `reference_solution_exposed=true`,
`verifier_tests_exposed=true`, and `held_out_evaluation=false`. Shared judges still
run; contamination findings are expected for this deliberate exposure and must
not be presented as a clean, held-out benchmark score. The official verifier and
reference patch are copied without modification from the pinned source task.

## Submit

From the repository root:

```bash
bash src/commit_utils/commit_deepswe_smoke.sh --dry-run
bash src/commit_utils/commit_deepswe_smoke.sh
```

The second command first caches a checksum-verified uv 0.9.7 binary and installs
the pinned CPU-only mini-swe agent environment on the submission node, then
submits exactly one GPU training/evaluation job. If the task
SIF is absent, it first submits a separate image preparation job and makes
the GPU job depend on its success. The GPU job has a configurable total walltime; the preparation job has 30 minutes. Image
preparation does not consume the training job's GPU budget. Never build images
inside the training container. A failed preparation cancels the dependent job.

Defaults: `module load singularity/3.9.7`, automatic partition routing, one A100, 8 CPU cores,
64 GiB RAM. The training agent gets 60 minutes (including CLI
update and any exploratory evaluations). Independent review and final evaluation
run afterward, outside that budget. This is a single-task integration test, not
a full DeepSWE score. The official task deadlines are preserved: 3 hours for the
model's task rollout and 30 minutes for the verifier.

Configuration is read only from this checkout's `.env`. There is no fallback
into another checkout. Copy `src/deepswe/config.env.example` to `.env` if needed
and set `OPENAI_API_KEY` privately there. The host passes this key through the relevant Codex authentication variables only to outer Codex and the independent judges, never to the model's shell or verifier.

Images, cache and outputs must resolve inside this repository or its explicitly
configured external storage roots. Paths into unrelated checkouts are rejected.
The default images are `containers/standard.sif`, `containers/vllm_debug.sif`, and
`containers/deepswe-happy-dom.sif`. Build/provision the first two before submission;
the task image is pulled automatically if absent. No image is borrowed from
another checkout.

```bash
# From PostTrainDeepSWEBench; configure its .env first.
DEEPSWE_ACCOUNT=conf-cvpr-2026.11.23-ghanembs bash src/commit_utils/commit_deepswe_smoke.sh
```

Overrides: `DEEPSWE_SINGULARITY_MODULE`, `DEEPSWE_BASE_MODEL`, `DEEPSWE_GRES`,
`DEEPSWE_ACCOUNT`, `DEEPSWE_PARTITION` (normally unset on Ibex), `HF_HOME`,
`DEEPSWE_RESULTS`, `DEEPSWE_TRAIN_SECONDS` (default 3600, maximum 3600),
`CODEX_CLI_VERSION`, `POST_TRAIN_BENCH_SKIP_CLI_UPDATE`.
The scaffold/chat template currently supports Qwen3 only. Build ML images on an
Ibex compute node with `module load singularity/3.9.7`, `--fakeroot`, and a local
`/tmp` directory for `SINGULARITY_TMPDIR`. Login nodes do not have the required
subordinate UID mappings. All new Slurm jobs use the conference account, including preparation and
verifier checks. Allocate a GPU even for CPU-only work to satisfy its policy:

```bash
sbatch --account=conf-cvpr-2026.11.23-ghanembs --gres=gpu:a100:1 --cpus-per-task=16 --mem=64G \
  --time=00:59:00 --output=logs/deepswe_build_%j.out --error=logs/deepswe_build_%j.err \
  dev_utils/deepswe/rebuild_standard.sh --from-eval
```

This snapshots the current recipe and requirements and produces a candidate at
`results/image_build_JOB_ID/standard.sif`, with recipe and image checksums.
`--from-eval` uses `containers/deepswe_standard.def`: it inherits the local fixed
evaluation SIF's CUDA/PyTorch stack, restores the standard training dependency
pins (including the regular Inspect AI release), and installs the standard Codex
CLI. It avoids downloading the same large CUDA libraries again. The base SIF is
hard-linked into the build snapshot and its checksum is recorded.
Without `--from-eval`, the script builds the CUDA-based `standard.def` from scratch
(allow a longer walltime). That mode replaces `--torch-backend=auto` with `cu129`
in the build copy because a CPU node would otherwise select CPU wheels.
The original standard recipe is preserved alongside either build.
It does not overwrite the active SIF. Editing a `.def` does not update an existing SIF;
inspect the embedded recipe with `singularity inspect --deffile IMAGE`.
The image is assembled in node-local `/tmp` and copied back only after the build
succeeds. Check `quota -s` before building: filesystem-wide `df` free space does
not reflect the user's home quota. Allow roughly 12 GiB for the candidate in
addition to the current image, plus room for compatibility-test artifacts.
Before replacing the active image, run `dev_utils/deepswe/check_image_compat.sh
CANDIDATE_PATH` in a 30-minute A100 Slurm allocation. It records package inventories,
checks direct dependency drift and core train/eval versions, and exercises a
synthetic tiny Qwen3 LoRA update, merge, safetensors export, tokenizer/template
round trip, matching Transformers logits, and vLLM generation in the fixed eval
image. No benchmark data or existing trained checkpoint is changed. These checks
validate the starting environment; agents can still change their own packages.
Temporary probe weights and container homes stay in node-local `/tmp`; only logs,
package inventories and comparison reports are retained in `results/image_compat_JOB_ID`.
Ibex's unprivileged `singularity pull docker://...` is sufficient for the task image.

## Execution and isolation

The training environment needs no host root privileges. `prepare_uv.sh` downloads the
pinned x86_64 Linux binary into `.cache/deepswe-tools/` (gitignored). Each run
copies it into its own `/home/ben/.local/bin`. Before starting Codex, the harness
creates `/home/ben/.venv` inside the training image with `--system-site-packages`,
checks training imports and GPU availability, and records `training_env.log`.
The virtual environment reuses the image's PyTorch and other training packages;
additional packages are installed with:

```bash
uv pip install --python /home/ben/.venv/bin/python PACKAGE
```

Both the launcher and Codex's shell environment policy select this Python and uv.
As in the original PostTrainBench prompt, installing or changing packages for
experimentation is allowed. No additional package-version constraints or Python
download restrictions are imposed. The base SIF remains immutable; use the
persistent writable virtual environment. New packages stay in the task's writable home;
the evaluation image remains independent, so checkpoints must load there without
those additions. Environment setup failures stop the job before any model API call.
To prepare the binary explicitly on a login node, run
`bash src/deepswe/prepare_uv.sh`; it is reused across submissions.

The host Python process owns all Singularity launches. The training container
reads its full prompt from `training_prompt.txt` through `start_agent.sh`;
multiline prompts must not be passed through Singularity 3.9's `--env`, which
truncates the value at the first newline. The shared Codex solve script then
receives `PROMPT` exported inside the container.
The training container
calls a loopback HTTP bridge for optional checkpoint evaluation; it never starts
nested containers. The bridge accepts checkpoints only below the training task
directory. A distinct private credential, unavailable to Codex, permits the
trusted evaluator's shell/verify callbacks. The model itself sees only the task
prompt, action protocol and command observations.

New runs use the official **mini-swe-agent 2.4.6** implementation, pinned to
`04d809ceab9df28f9adaed044884180159172930`, with the upstream `mini.yaml` prompts
and native `bash` tool calls. This follows DeepSWE's Pier/mini-swe agent choice;
it is not a claim to reproduce its Modal leaderboard configuration. The Ibex
adapter replaces the environment transport, while Inspect retains scoring and
evaluation status. The original ML SIF serves the checkpoint through a private
loopback vLLM endpoint using the existing Qwen3 chat template and `hermes` tool
parser. No ML dependency/image rebuild is required. CPU-only mini dependencies
live at `$DEEPSWE_STORAGE_RESULTS/mini-swe-venv`, pinned by
`mini_swe/requirements.lock`; prepare them with `bash src/deepswe/prepare_mini.sh`.

The output cap is `model.model_kwargs.max_tokens`, passed from the existing
`evaluate.py --max-tokens 4000` interface. Pier's corresponding agent kwarg is
`model_kwargs={"max_tokens":4000}`, which it translates to mini's
`-c model.model_kwargs.max_tokens=4000`. The vLLM total input-plus-output window
is 32768 tokens. **Native mini-swe does not automatically summarize.** Context
exhaustion ends the rollout and grades the existing committed diff once; its
exit reason is recorded. Authentication/server failures produce no score.
Native consecutive-format-error handling is retained, with no custom turn cap.
Each API call receives its own seed derived through `per_sample_seed.py`.

Training receives `mini_swe_config.yaml`, `mini_swe_bash_tool.json` and
`evaluation_protocol.txt`. It must train native tool-call trajectories with the
tool schema supplied to the chat template. Completion uses the bash command
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` after committing the changes.
Existing runs whose host-owned `run_config.json` lacks a harness field retain the
legacy JSON protocol; changing code must not silently change their interface.

Task execution uses the official prebuilt image from its `task.toml`. A fresh
copy of `/app` is writable for the rollout. Only `/app` and its task home persist
between commands. The task is offline (`--net --network none`), receives no GPU
or provider credentials, and has no hidden test mount. The model must commit its
changes. The host collects `git diff --binary BASE HEAD`, matching the upstream
collect contract, and starts a separate clean workspace for verification. Only
the verifier receives the original tests and read-only collected patch. The
original `tests/test.sh` and `grader.py` are unmodified; binary reward, pass-to-pass
and fail-to-pass results are preserved. The reference solution is available only
to the teacher in the explicitly authorized teaching experiment.

Shell observations merge the command's stdout and stderr **inside** the task
container. Singularity launcher diagnostics are saved separately in each
evaluation's `singularity-shell.log`, so they cannot hide mini-swe's first-line
submission marker. Compiler/test stderr still reaches the model. Regression
coverage is in `test_shell_transport.py`; `test_shell_transport.sh` runs the same
submission check through native mini-swe and the real task SIF in a CPU Slurm job.

All DeepSWE containers (training agent, model inference, task and verifier) use
`--fakeroot` on Ibex compute nodes, where the
account has subordinate UID/GID mappings. Login nodes may lack these mappings.
This preserves the complete repository, including root-only `.pdm-build`, without
excluding files or changing tests. Slurm entry points use job-local `/tmp` for
Singularity's temporary image extraction; shared-filesystem extraction is slow.
Processes see UID 0 inside; on the host they remain mapped to the submitting user.
This is an explicit DeepSWE runtime choice; the original PostTrain runners are unchanged.

To evaluate an existing checkpoint again without training or calling Codex:

```bash
bash src/commit_utils/commit_deepswe_smoke_eval.sh results/deepswe_smoke/53908391
```

This submits one A100 job with a configurable final-evaluation walltime and saves results beneath
the source run in `reeval_JOB_ID/`. The original checkpoint is mounted read-only
and the original run's logs and status are retained. Before deploying changes to
task isolation, run `dev_utils/deepswe/test_task_fakeroot.sh` in a CPU Slurm job
to check full copying, network/credential isolation, persistence and fresh workspaces.

This is a native Singularity adapter for **this single Harbor-format task**, not
a generic Pier backend. Other tasks may require extra writable paths, services,
network policies or collect hooks and must be adapted before being added. The
current Ibex configuration only has `/etc/hosts` and `/etc/localtime` admin bind
paths; custom installations must check admin binds do not expose held-out files.

## Outputs and limitations

- `logs/deepswe_<job>.{out,err}`: Slurm output.
- `results/deepswe_smoke/<job>/run_config.json`, `status.json`: configuration and status.
- `training_home/task/final_model/`: submitted model, with tokenizer and full weights.
- `solve_out.txt`, `prompt.txt`, `training_home/cli_version.txt`: training agent provenance.
- `eval_*/`: Inspect trajectories, committed patch, fresh workspaces, verifier logs/reward.
- `metrics.json`: only written after successful final evaluation; uses the shared per-seed aggregation schema.

Training is capped at 60 minutes, including exploratory evaluations. After training,
the trace is parsed and the shared `run_all_judges` runs all four judge categories
(including all three contamination slots), then `run_final_eval.sh` runs the
single seed 72332. Judge model, reasoning effort, CLI pins and policies come from
the shared judge configurations; they are independent of the training agent's low
effort. Missing verdicts retain the upstream warning/continue behavior and must
not be interpreted as passing an audit.

The final evaluation uses the shared 4/3/2 retry cascade at 4000/3000/2000 output
tokens. A successful score of zero is retained, not retried as an infrastructure
failure. Sampling uses the model defaults. There is no custom 12-turn cap, 8192
context cap or manual history truncation; the new mini-swe server uses the
model's 32768-token total window. The task's agent and verifier deadlines
are 10800 and 1800 seconds. These are evaluation deadlines, outside the 60-minute
training budget.

`DEEPSWE_JOB_WALLTIME` and `DEEPSWE_EVAL_WALLTIME` both default to `01:30:00`.
Training remains capped at 60 minutes. Review and final evaluation share the time
remaining in the 90-minute allocation; this budget can interrupt an unfinished
review or rollout. The pipeline does not silently extend the Slurm request or
turn incomplete evaluation into a zero score.
Use the dedicated commit script rather than the generic Condor submission.
The directory name `deepswe_smoke` is retained for existing paths; this remains a
single-task experiment, not a full DeepSWE benchmark result.

A checkpoint must be a complete Hugging Face model and tokenizer; LoRA adapters
must be merged. The active prompt is the special teaching experiment described below; the
ordinary shared prompt still prohibits test-derived training data.
No minimum score or positive training effect is assumed for a 60-minute training run.

## Validation

```bash
python3 -m unittest discover -s dev_utils/deepswe -p test_runtime.py -v
python3 -m compileall -q src/deepswe src/eval/tasks/deepswe
bash -n src/commit_utils/commit_deepswe_smoke.sh src/deepswe/*.sh
module load singularity/3.9.7
# The following image tests require a compute-node allocation, not a login node.
# Set DEEPSWE_CONTAINER_TMPDIR to a job-owned temporary directory under /tmp.
python3 dev_utils/deepswe/test_prompt_transport.py
python3 dev_utils/deepswe/test_training_env.py
```

`dev_utils/deepswe/test_inspect.py` additionally exercises the real Inspect solver
and scorer API for legacy checkpoints in the evaluation image (no GPU/API).
Run `$DEEPSWE_MINI_PYTHON dev_utils/deepswe/test_mini.py` for the native agent's
tool protocol, max-token/seed transport, upstream prompt parity, bounded format
errors, and context-exhaustion scoring. `test_mini_gpu.sh CHECKPOINT` exercises
the actual vLLM API, task image and separate verifier in a short GPU allocation;
an older checkpoint used for that smoke test is not a comparable model score.
`test_training_env.py` requires the cached uv binary and checks fakeroot setup,
Codex shell PATH, inherited imports, and an offline package installation into the
virtual environment, without a GPU or model API.
`test_gpu_fakeroot.py` checks CUDA tensor operations in both ML images and the
training virtual environment under fakeroot; it requires one allocated GPU.

Sources: [Ibex](https://docs.hpc.kaust.edu.sa/systems/ibex/),
[Singularity on Ibex](https://docs.hpc.kaust.edu.sa/soft_env/prog_env/containers/mpi.html),
[Inspect solvers](https://inspect.aisi.org.uk/solvers.html).
Vendored task: `datacurve-ai/deep-swe`, commit
`0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`, Apache-2.0, license retained in
`src/eval/tasks/deepswe_smoke/happy_dom_task/LICENSE`. Upstream tests are unchanged;
the Happy DOM reference solution is included only under
`src/deepswe/teaching_materials/happy_dom/solution` for the explicitly authorized
teaching experiment. The older FastAPI task files are retained for provenance.

### Ibex storage

This checkout's `.env` sets `DEEPSWE_STORAGE_RESULTS` to
`/ibex/user/yangz0h/results/post_train_deepswe` and `DEEPSWE_STORAGE_IMAGES` to
`/ibex/user/yangz0h/datasets/post_train_deepswe`. Training runs live in
`$DEEPSWE_STORAGE_RESULTS/deepswe_smoke`, model caches in
`$DEEPSWE_STORAGE_RESULTS/.cache/huggingface`, and Singularity download caches in
`$DEEPSWE_STORAGE_RESULTS/.cache/singularity-build`. Active SIFs live directly
in the image directory; historical SIFs are under `archive/`, new builds under
`build_JOB_ID/`. Build reports and compatibility reports remain in the results
storage. Build/extraction temporary files still use compute-node `/tmp`.

Local `results`, `.cache`, and `containers/*.sif` symlinks preserve old paths.
The repository and credentials stay in this checkout; no sibling checkout is
used. Without explicit storage configuration, paths default to this checkout.

### Framework parity

The ordinary prompt remains available via `make_general_prompt`. The active
teaching experiment uses its own explicit prompt. The launcher uses shared templates, trace parser,
system monitor, judge policies and final-evaluation retry/aggregation code.
Hugging Face cache writes use a private FUSE overlay, as in the original runner;
the shared cache is only the read-only lower layer. The FUSE mountpoint is on
node-local /tmp because Ibex forbids FUSE mounts on its shared filesystem;
private cache data remains in the configured results storage. Training cache changes are
not carried into fixed-image evaluation.

Remaining intentional differences are the Slurm/module/Singularity/fakeroot
launcher, writable Python venv needed on Ibex, configured external storage,
the DeepSWE host bridge and separate task/verifier containers, and the requested
60-minute training / 90-minute Slurm / one-task / one-seed / gpt-6-sol-low experiment,
and the explicitly requested teaching prompt with reference and test exposure. The task
verifier itself is unchanged. This dedicated experiment's result directory
layout is retained; it is not registered as a scored full benchmark in the
upstream collection or Modal task generator.

### Happy DOM verifier controls

Before launching training for `happy-dom-abort-pending-body-reads`, run
`dev_utils/deepswe/test_happy_dom_verifier.sh` through Slurm using the conference
account and one GPU allocation. The check uses the unmodified upstream verifier
on four independent repositories: untouched base (must score 0), the official
solution (must score 1), that solution plus a syntax error (must score 0), and
that solution plus a module-load exception (must score 0). It also checks raw
JUnit failure counts and rejects nonzero fail-to-pass credit for broken imports.
Reports are saved in `happy_dom_verifier_check_JOB_ID/summary.json` under the
configured result storage. Do not start training if these controls fail.
Historical true-myth task files and results remain available for provenance.

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
  --error=logs/deepswe_qwen35_prepare_%j.err src/deepswe/prepare_full_job.sh
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
