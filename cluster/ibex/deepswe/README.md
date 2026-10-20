# DeepSWE on Ibex

This directory contains only cluster setup and orchestration. Benchmark metadata,
prompts, protocols and evaluation code live alongside the other evaluation tasks:

- [deepswe](../../../src/eval/tasks/deepswe/README.md): all 113 held-out tasks, Qwen3.5-9B.
- [deepswe_smoke](../../../src/eval/tasks/deepswe_smoke/README.md): one-hour Qwen3.5 held-out smoke test (legacy teaching artifacts retained).

Configure the repository-root `.env` using `example.env`. `common.sh` reads it
without sourcing the Condor module setup. Model/image/result caches remain at
the configured external paths; moving this code does not move cached assets.
GPU jobs use account `conf-cvpr-2026.11.23-ghanembs`. CPU-only checks and aggregation
use `pi-ghanembs` without a GPU request.

Preparation: `prepare_full_job.sh` caches the model and suite, builds the dedicated
image and runs the GPU gate before activation; `prepare_images.sh` prepares smoke
images. `prepare_uv.sh` and `prepare_mini.sh` install the host helpers.

Submission commands, run from the repository root:

```bash
bash src/commit_utils/commit_deepswe.sh --dry-run
bash src/commit_utils/commit_deepswe_smoke.sh --dry-run
```

`full_job.sh` dispatches training, final-evaluation array tasks and aggregation.
`job.sh` runs smoke; `reevaluate_job.sh` reevaluates an existing smoke checkpoint.
`run.py` assembles the training sandbox, invokes Codex, then calls the shared
judges through `src/judges/deepswe/`. `training_support.py` composes the original
PostTrainBench prompt with task instructions and Ibex environment adaptations.
`setup_training_env.sh` supplies the same writable virtual environment in both modes.

Before a multi-GPU training run, submit `check_multi_gpu_job.sh` with the same
GPU allocation (currently two A100s). It checks the actual training environment,
CUDA allocation on both devices, NCCL all-reduce, and one DDP optimizer step
with synchronized weights. Its `ready.json` is written only after all checks pass.
Use `commit_deepswe.sh --after-validation JOB_ID` to make training depend on
successful validation; training also reruns the full-suite readiness check.

Singularity 3.9 parses `--env` arguments as comma-separated entries. Preserve
Slurm's `CUDA_VISIBLE_DEVICES` using `--env-file`; never replace it with guessed
device IDs. GPU count is taken from the allocation and checked inside the
container before starting the agent. The agent receives the allocated count in
`NUM_GPUS` and its prompt. Keep preflight and training launch settings aligned,
and do not edit probe code while its job is running.

Official references used for this workflow:

- [Ibex GPU jobs](https://docs.hpc.kaust.edu.sa/soft_env/job_schd/slurm/ibex_gpu_jobs.html)
- [Ibex container examples](https://docs.hpc.kaust.edu.sa/soft_env/prog_env/containers/runtime_examples.html)
- [Ibex PyTorch DDP](https://docs.hpc.kaust.edu.sa/soft_env/science_platforms/data_science/dist_mldl/torch_ddp.html)
- [Ibex job dependencies](https://docs.hpc.kaust.edu.sa/soft_env/job_schd/slurm/dependencies.html)
- [Singularity 3.9 environment handling](https://docs.sylabs.io/guides/3.9/user-guide/environment_and_metadata.html)

The old `src/deepswe/common.sh` and `prepare_full.py` are compatibility forwards
for preparation job 54006896, whose script Slurm already spooled before migration.
They contain no duplicate settings or implementation and can be removed after
that job finishes. New submissions use this directory exclusively.
