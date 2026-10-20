"""Compatibility entry point for preparation job 54006896."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2] / 'cluster/ibex/deepswe/prepare_full.py'), run_name='__main__')
