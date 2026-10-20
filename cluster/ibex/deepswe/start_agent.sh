#!/bin/bash
# Singularity 3.9 truncates multiline --env values. Read the prompt only after
# entering the container, so the shared Codex solve.sh receives the full text.
set -euo pipefail
PROMPT="$(cat /home/ben/training_prompt.txt)"
[[ -n "$PROMPT" ]] || { echo 'Training prompt is empty' >&2; exit 1; }
export PROMPT
exec bash /home/ben/agent_solve.sh
