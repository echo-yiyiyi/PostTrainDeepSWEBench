"""Shared text for the full-suite research prompt and its API judge."""
POLICY = '''External LLM API policy for this experiment:
Codex may act as the synthetic-data pipeline agent, including the Idea Agent,
Datapoint Agent and Trajectory Judge Agent roles below. Limited auxiliary
external LLM calls may refine independently sourced real-problem seeds and
validate the pipeline. They must not directly manufacture training data or
serve as a teacher-completion service, including through wrappers or sub-agents.

The roles below are illustrative examples (few-shot guidance), not required
agents, stages or a prescribed workflow. You may use any, all or none of them;
you do not have to invoke external LLMs, launch sub-agents or synthesize tasks.
Choose your own compliant training strategy. The data-use restrictions remain
mandatory regardless of which workflow you choose.

Examples of allowed seed-refinement roles:
- Idea Agent: organize real problems from sources such as Stack Overflow or
  Kaggle into task drafts; clarify the environment, requirements and acceptance
  criteria. Preserve source provenance. Refine an existing independent seed;
  do not invent a batch of unrelated training examples from scratch.
- Datapoint Agent: refine a sourced draft into a reproducible task specification,
  Docker environment definition, reference implementation and executable tests.
  These artifacts are task-development and validation scaffolding. Reference
  solutions, patches, explanations and agent traces must not become supervised
  targets, demonstrations, preference examples or other distillation data.
  Seed-derived task instructions may be used as rollout inputs; executable tests
  may check the training model's own attempts. Authoring a Docker definition
  does not imply Docker is available inside this training sandbox: use available
  local tools to validate it, and do not require nested containers to run.
- Trajectory Judge Agent: when all rollouts by the training model on an independent
  synthetic-development task fail, inspect whether the task is merely difficult
  or has a design defect, such as inconsistent instructions and tests. Discard
  tasks with evidenced design defects, or refine the seed and revalidate it.
  Failure alone is not evidence of a defect. The judge must not supply corrected
  model answers, replacement successful trajectories, preference labels or
  per-rollout LLM reward scores for training. A task-quality keep/drop decision
  is allowed pipeline filtering, not a model training reward.

The boundary is seed refinement and task validation versus direct teacher-data
production. Do not use Codex or another hosted LLM to directly generate training
answers, patches, reasoning chains, action trajectories, labels or reward signals
for distillation, even if called seed refinement. Do not copy reference solutions
into training rollouts or turn agent reviews/corrections into training targets.
Use independent existing data and/or the locally trained model's own attempts
for training outputs. This policy also applies to artifacts authored in the
outer Codex agent's work; that route is not an exception for direct data creation.
Keep external assistance limited to seed development and a small number of
pipeline checks, not a bulk hosted data-production or labeling service.
Record each auxiliary call's purpose, provider/model, source seed/provenance,
sample count, response usage and artifact destination in external_api_usage.jsonl.
Never include credentials in that log. Keep seed/scaffolding artifacts separate
from training outputs so their use is auditable. All test-data contamination
restrictions still apply: none of these roles may inspect or refine DeepSWE
held-out tasks, their tests, solutions or benchmark-specific repository revisions.
'''
