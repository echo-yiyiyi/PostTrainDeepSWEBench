# DeepSWE post-training smoke test on Ibex

One **outer Codex `gpt-6-sol` training run with reasoning effort `low`**, fine-tuning
`Qwen/Qwen3-1.7B-Base`, followed by one evaluation of that checkpoint on the
`fastapi-implicit-head-options` DeepSWE task. Codex is the training agent; the
trained Qwen model is the agent operating on the evaluation repository.
This is not a direct evaluation of Codex's repository-solving ability.

## Submit

From the repository root:

```bash
bash src/commit_utils/commit_deepswe.sh --dry-run
bash src/commit_utils/commit_deepswe.sh
```

The second command first caches a checksum-verified uv 0.9.7 binary locally on
the submission node, then submits exactly one GPU training/evaluation job. If the task
SIF is absent, it first submits a separate CPU image preparation job and makes
the GPU job depend on its success. The GPU job has a 59-minute walltime; the CPU preparation job has 30 minutes. Image
preparation does not consume the training job's GPU budget. Never build images
inside the training container. A failed preparation cancels the dependent job.

Defaults: `module load singularity/3.9.7`, automatic partition routing, one A100, 8 CPU cores,
64 GiB RAM, 59 minutes total. The training agent gets 40 minutes (including CLI
update and any exploratory evaluations), leaving time for model loading,
rollout and the original verifier. This tests the plumbing, not convergence or
full DeepSWE performance. The single task's official verifier allows 30 minutes
by itself; this smoke harness caps it at 4 minutes and the entire evaluation at
12 minutes. Timeout/infrastructure failure produces no fabricated zero score.

Configuration is read only from this checkout's `.env`. There is no fallback
into another checkout. Copy `src/deepswe/config.env.example` to `.env` if needed
and set `OPENAI_API_KEY` privately there. The host passes this key as
`CODEX_API_KEY` only to outer Codex, never to the model's shell or verifier.

All images, cache and outputs must resolve inside this repository; external paths
and symlinks into another checkout are rejected, including stale exported values.
The default images are `containers/standard.sif`, `containers/vllm_debug.sif`, and
`containers/deepswe-fastapi.sif`. Build/provision the first two before submission;
the task image is pulled automatically if absent. No image is borrowed from
another checkout.

```bash
# From PostTrainDeepSWEBench; configure its .env first.
DEEPSWE_ACCOUNT=conf-cvpr-2026.11.23-ghanembs bash src/commit_utils/commit_deepswe.sh
```

Overrides: `DEEPSWE_SINGULARITY_MODULE`, `DEEPSWE_BASE_MODEL`, `DEEPSWE_GRES`,
`DEEPSWE_ACCOUNT`, `DEEPSWE_PARTITION` (normally unset on Ibex), `HF_HOME`,
`DEEPSWE_RESULTS`, `DEEPSWE_TRAIN_SECONDS` (default 2400, maximum 2400),
`CODEX_CLI_VERSION`, `POST_TRAIN_BENCH_SKIP_CLI_UPDATE`.
The scaffold/chat template currently supports Qwen3 only. To build the local ML
SIFs, run `singularity build containers/standard.sif containers/standard.def` and
`singularity build containers/vllm_debug.sif containers/vllm_debug.def` on a build
host with suitable privileges, after loading the Singularity module. Ibex's
unprivileged `singularity pull docker://...` is sufficient for the task image.

## Execution and isolation

The training environment needs no root access. `prepare_uv.sh` downloads the
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
download restrictions are imposed. System directories are read-only; use the
writable virtual environment. New packages stay in the task's writable home;
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

The evaluator uses Inspect AI + vLLM to load `final_model`, Qwen3 chat formatting,
greedy decoding, per-call seeds from `per_sample_seed.py`, and at most 12 JSON
shell actions. This deliberately small scaffold is a smoke-test configuration,
not the official Pier/mini-swe-agent leaderboard configuration.

Task execution uses the official prebuilt image from its `task.toml`. A fresh
copy of `/app` is writable for the rollout. Only `/app` and its task home persist
between commands. The task is offline (`--net --network none`), receives no GPU
or provider credentials, and has no hidden test mount. The model must commit its
changes. The host collects `git diff --binary BASE HEAD`, matching the upstream
collect contract, and starts a separate clean workspace for verification. Only
the verifier receives the original tests and read-only collected patch. The
original `tests/test.sh` and `grader.py` are unmodified; binary reward, pass-to-pass
and fail-to-pass results are preserved. No reference solution is vendored.

Task and verifier containers use `--fakeroot` on Ibex compute nodes, where the
account has subordinate UID/GID mappings. Login nodes may lack these mappings.
This preserves the complete repository, including root-only `.pdm-build`, without
excluding files or changing tests. Slurm entry points use job-local `/tmp` for
Singularity's temporary image extraction; shared-filesystem extraction is slow.
Training and model inference containers continue to run as the submitting user.

To evaluate an existing checkpoint again without training or calling Codex:

```bash
bash src/commit_utils/commit_deepswe_eval.sh results/deepswe_smoke/53908391
```

This submits one A100 job with a 30-minute walltime and saves results beneath
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
- `metrics.json`: only written after successful final evaluation; includes `smoke_only=true`.

The long reward-hacking judge suite is **not run** in this short smoke path.
Do not treat these outputs as audited PostTrainBench results or feed them into
its official aggregation. The ordinary Condor runner, other benchmarks and its
judge requirements remain unchanged. Use the dedicated commit script, not
`commit.sh` or the generic `run_final_eval.sh`, for this benchmark.

A checkpoint must be a complete Hugging Face model and tokenizer; LoRA adapters
must be merged. The agent prompt requires independent training data and prohibits
using this held-out task or its derivatives for training. The test prompt is
provided only as input to the existing contamination checker, matching PostTrain.
No minimum score or positive training effect is assumed for a 40-minute training run.

## Validation

```bash
python3 -m unittest discover -s dev_utils/deepswe -p test_runtime.py -v
python3 -m compileall -q src/deepswe src/eval/tasks/deepswe
bash -n src/commit_utils/commit_deepswe.sh src/deepswe/*.sh
module load singularity/3.9.7
python3 dev_utils/deepswe/test_prompt_transport.py
python3 dev_utils/deepswe/test_training_env.py
```

`dev_utils/deepswe/test_inspect.py` additionally exercises the real Inspect solver
and scorer API in the existing evaluation image with a mock model (no GPU/API).
`test_training_env.py` requires the cached uv binary and checks non-root setup,
Codex shell PATH, inherited imports, and an offline package installation into the
virtual environment, without a GPU or model API.

Sources: [Ibex](https://docs.hpc.kaust.edu.sa/systems/ibex/),
[Singularity on Ibex](https://docs.hpc.kaust.edu.sa/soft_env/prog_env/containers/mpi.html),
[Inspect solvers](https://inspect.aisi.org.uk/solvers.html).
Vendored task: `datacurve-ai/deep-swe`, commit
`0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea`, Apache-2.0, license retained in
`src/eval/tasks/deepswe/harbor_task/LICENSE`. Upstream tests are unchanged;
`solution/` is intentionally excluded.
