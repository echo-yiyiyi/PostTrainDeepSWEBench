"""Host-only final evaluator for the full-suite Slurm protocol.

Usage: evaluate_final_eval.py RUN --index TASK_INDEX [--seed 72332]
       evaluate_final_eval.py RUN --aggregate
The agent-facing evaluate.py exposes aggregate training-trial scores via the host bridge.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'deepswe'))
from full_evaluate import main

if __name__ == '__main__':
    main()
