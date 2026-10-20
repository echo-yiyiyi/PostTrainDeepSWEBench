"""Legacy teaching-task selection and shared agent protocol; current smoke uses the host manifest."""

TASK_ID = 'happy-dom-abort-pending-body-reads'
TASK_DIRECTORY = 'happy_dom_task'
TEACHING_NAME = 'happy_dom'
BASE_COMMIT = '82a0888cb2c87a6123e05424b528f8e8c9b3e426'
TASK_IMAGE = 'public.ecr.aws/d3j8x8q7/swe-bench-202605:kh7c2re7cvbseq7xz6samd1xr182y1dc-v1.1'
EXPERIMENT_MODE = 'task_specific_training'

PROTOCOL = '''Evaluation uses the official mini-swe-agent v2 agent and mini.yaml
prompt templates. Read mini_swe_config.yaml for the exact system/instance and
observation/error templates, and mini_swe_bash_tool.json for the tool schema.
The model emits native bash function calls, NOT {"command":...}/{"done":true}
standalone JSON actions. Use templates/qwen3.jinja with tools=[the bash schema]
when preparing training inputs. Assistant tool calls use <tool_call> blocks;
tool observations are rendered with <tool_response> by the same chat template.
Commands run in /app in independent shells; repository edits persist, shell
variables and cd do not. Commit all changes before submitting. Submit via the
bash tool: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT (alone).
max_tokens controls output per API request via model.model_kwargs.max_tokens.
The server overrides only generation_config.max_new_tokens with this budget,
so a checkpoint cannot silently impose a smaller output limit.
The server's total input-plus-output context is max_model_len. Native mini-swe
does not automatically summarize; context exhaustion ends the rollout, and the
existing committed diff is graded by the official separate verifier.
'''
