#!/bin/bash
# Shared PostTrain judge policies; only the process launcher is Ibex-specific.
set -euo pipefail
source cluster/ibex/deepswe/common.sh
source src/judges/judge_lib.sh
result="$(realpath -- "$1")"
judge_home="$result/judge_home"
mkdir -p "$judge_home/task" "$judge_home/tmp"
copy_task_for_judge "$result/training_home/task" "$judge_home/task"
cp "$result/solve_parsed.txt" "$judge_home/solve_parsed.txt"
benchmark="$(python3 -c 'import json,sys; print("deepswe" if json.load(open(sys.argv[1])).get("experiment_mode") == "full_held_out" else "deepswe_smoke")' "$result/run_config.json")"
prepare_judge_sandbox "$judge_home" "$benchmark" "$result/training_home/task/final_model/config.json"
if [[ "$benchmark" == deepswe ]]; then
    python3 - "$result/suite_manifest.json" "$judge_home/test_data.json" <<'PYREF'
import json,sys
from pathlib import Path
suite=json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps([{'id':t['id'], 'question':(Path(t['path'])/'instruction.md').read_text()} for t in suite['tasks']]))
PYREF
else
    cp "$result/training_home/test_data.json" "$judge_home/test_data.json"
fi
JUDGE_RUNTIME=local
setup_judge_codex_auth "$judge_home"
# The shared local helper sees host paths; Codex runs at /home/ben on Ibex.
sed -i "s|$judge_home/task|/home/ben/task|g" "$judge_home/.codex/config.toml"
# Keep judge selection, prompts, model/CLI pins, slots, parsing and verdicts in
# judge_lib.sh. A separate Singularity home replaces its Modal-local launcher.
run_judge_exec() {
    local job_dir="$1" job_tmp="$2" output_json="$3" prompt="$4"
    printf '%s' "$prompt" > "$job_dir/judge_prompt.txt"
    DEEPSWE_JUDGE_MODEL="$JUDGE_MODEL" DEEPSWE_JUDGE_EFFORT="$JUDGE_REASONING_EFFORT" \
    DEEPSWE_JUDGE_CLI="$JUDGE_CODEX_VERSION" python3 src/judges/deepswe/judge_exec.py \
        "$job_dir" "$output_json"
}
model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_to_train"])' "$result/run_config.json")"
run_all_judges "$judge_home" "$judge_home/tmp" "$result" "$benchmark" "$model" codex gpt-6-sol
# Record completeness separately from model accuracy. An absent verdict is not a pass.
expected=()
for judge in "${ALL_JUDGES[@]}"; do
    load_judge_conf "$judge"
    expected+=("judgement_${JUDGE_OUTPUT_ID}.json")
    if [[ "$judge" == data_contamination_judge ]]; then
        for ((slot=2; slot<=CONTAMINATION_JUDGE_SLOTS; slot++)); do
            expected+=("judgement_multi_runs/judgement_${JUDGE_OUTPUT_ID}_run${slot}.json")
        done
    fi
done
python3 - "$result" "${expected[@]}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
missing=[]
for name in sys.argv[2:]:
    try:
        value=json.loads((root/name).read_text())
        if not isinstance(value,dict):
            missing.append(name)
    except (OSError,ValueError):
        missing.append(name)
(root/'judges_status.json').write_text(json.dumps({'complete':not missing,'missing_or_invalid':missing},indent=2)+'\n')
PY
