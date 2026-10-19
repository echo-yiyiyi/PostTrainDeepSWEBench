# Pinned upstream mini-swe-agent assets

`mini.yaml`, `bash_tool.json` and `LICENSE.md` come from mini-swe-agent 2.4.6,
revision `04d809ceab9df28f9adaed044884180159172930`:
https://github.com/SWE-agent/mini-swe-agent/tree/04d809ceab9df28f9adaed044884180159172930
The JSON schema is exported unchanged from
`minisweagent.models.utils.actions_toolcall.BASH_TOOL`.

DeepSWE's documented agent choice is Pier + mini-swe-agent. We checked Pier's
adapter at revision `4d3c14041d16443f3f9f460dcdf23629994a304e`, specifically
`src/pier/agents/installed/mini_swe_agent.py`: its default is `mini.yaml`,
`cost_limit=0`, autonomous mode, and immediate exit after submission. It does
not pin a mini version by default, so this checkout explicitly pins one.
The historical leaderboard's exact agent version is not asserted here.

Its `model_kwargs` agent argument forwards each field as a mini CLI override.
For example, `model_kwargs={"max_tokens":4000}` becomes:

```bash
mini-swe-agent --yolo --exit-immediately -c mini.yaml \
  -c model.model_class=litellm \
  -c model.model_kwargs.max_tokens=4000
```

For a local OpenAI-compatible vLLM server, use `LitellmModel` (Chat Completions),
not Pier's automatic `openai/` routing to the Responses API. Our Python entry
point sets the same `model.model_kwargs` fields. `max_tokens` is a per-response
output cap; there is no automatic context summarization in this pinned agent.
The existing PostTrain `evaluate.py --max-tokens` option sets this field.

Only Ibex environment transport and per-call deterministic sampling seeds are
adapted. The native agent and model own tool parsing, error messages, prompt
formatting, trajectory saving, and completion detection. `requirements.lock`
pins the CPU agent environment independently of the unchanged training/vLLM
image packages. No teacher or cloud-model credentials enter this environment.
