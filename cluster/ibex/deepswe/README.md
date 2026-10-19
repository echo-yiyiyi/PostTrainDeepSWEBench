# DeepSWE on Ibex

This directory contains only cluster setup and orchestration. Benchmark metadata,
prompts, protocols and evaluation code live alongside the other evaluation tasks:

- [deepswe](../../../src/eval/tasks/deepswe/README.md): all 113 held-out tasks, Qwen3.5-9B.
- [deepswe_smoke](../../../src/eval/tasks/deepswe_smoke/README.md): single-task teaching/debug experiment.

Configure the repository-root `.env` using `example.env`. `common.sh` reads it
without sourcing the Condor module setup. Model/image/result caches remain at
the configured external paths; moving this code does not move cached assets.
All Slurm jobs use account `conf-cvpr-2026.11.23-ghanembs` and allocate a GPU.

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

The old `src/deepswe/common.sh` and `prepare_full.py` are compatibility forwards
for preparation job 54006896, whose script Slurm already spooled before migration.
They contain no duplicate settings or implementation and can be removed after
that job finishes. New submissions use this directory exclusively.
