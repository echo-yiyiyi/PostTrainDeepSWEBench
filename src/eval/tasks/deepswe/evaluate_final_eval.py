"""Trusted entry point, invoked by the host bridge in a fresh ML container.

The generic Condor run_final_eval.sh is not this benchmark's launcher: use
src/commit_utils/commit_deepswe.sh, which owns the external task environments.
"""
import runpy

if __name__ == '__main__':
    runpy.run_path('/harness/inspect_eval.py', run_name='__main__')
