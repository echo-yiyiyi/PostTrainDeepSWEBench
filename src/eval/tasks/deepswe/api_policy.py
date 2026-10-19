"""Shared text for the full-suite research prompt and its API judge."""
POLICY = '''External LLM API policy for this experiment:
Do not use hosted LLMs as teachers to directly generate training examples,
answers, patches, reasoning chains, action trajectories, preference labels or
reward signals for distillation, including through wrappers or sub-agents.
Codex may act as the synthetic-data pipeline agent: design, author, execute,
and debug the pipeline, and write synthetic examples as part of that agent's
own work. This permission is not a ban on Codex acting inside the pipeline.
Do not instead use per-example Codex/LLM calls as a bulk teacher-completion
service whose answers or trajectories become distillation targets.
The research agent may make a small number of auxiliary external LLM calls to
validate the synthetic-data pipeline, inspect a few examples, debug processing
code, or assist with non-generative data processing. These are development
checks, not a bulk data-production or labeling service. Do not turn the returned
answers, corrections, solutions, or judgments into model training targets or
training rewards. Record each auxiliary call's purpose, provider/model, sample
count, response usage and destination in external_api_usage.jsonl. Never include
credentials in that log. The training agent's own authored examples and local
model inference are allowed. All test-data contamination restrictions still apply.
'''
