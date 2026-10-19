"""Real-container verifier controls, before spending a GPU training budget."""
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, 'src/eval/tasks/deepswe')
from runtime import ROOT, TaskEnvironment, container, run
from evaluation_config import TASK_ID, TEACHING_NAME


def main():
    assert TASK_ID == 'happy-dom-abort-pending-body-reads'
    output = Path(os.environ['DEEPSWE_STORAGE_RESULTS']) / (
        'happy_dom_verifier_check_' + os.environ['SLURM_JOB_ID'])
    output.mkdir()
    summary = {'task_id': TASK_ID, 'model_evaluation': False, 'checks': {}}
    solution = ROOT / 'src/eval/tasks/deepswe_smoke/task_context/teaching_materials' / TEACHING_NAME / 'solution'
    for name, expected in [('baseline', 0), ('oracle', 1), ('syntax_error', 0), ('import_error', 0)]:
        directory = output / name
        task = TaskEnvironment(directory, Path(os.environ['DEEPSWE_TASK_SIF']))
        if name != 'baseline':
            run(container(task.image, task.home,
                          [f'{task.app}:/app', f'{solution}:/solution:ro'], pwd='/app')
                + ['bash', '/solution/solve.sh'], timeout=120)
        if name in ('syntax_error', 'import_error'):
            # Break a source module imported by both base and new tests, after
            # applying the correct answer. Only commits reach the verifier.
            source = task.app / 'packages/happy-dom/src/fetch/Request.ts'
            error = ('\nexport const verifierSyntaxError = ;\n' if name == 'syntax_error'
                     else '\nthrow new Error("verifier control: module load failed");\n')
            source.write_text(source.read_text() + error)
            result = task.shell('git add packages/happy-dom/src/fetch/Request.ts && '
                                'git commit -m "Verifier negative control"')
            if result['exit_code'] != 0:
                raise RuntimeError(result)
        reward = task.verify()
        reports = {}
        for report in (directory / 'verifier/reports').glob('*.xml'):
            tree = ET.parse(report).getroot()
            reports[report.name] = dict(tree.attrib)
        ok = reward['reward'] == expected and not reward.get('apply_failed')
        failures = sum(int(r.get('failures', 0)) + int(r.get('errors', 0))
                       for r in reports.values())
        if name == 'baseline':
            ok = ok and reward['p2p_passed'] == reward['p2p_total']
        elif name == 'oracle':
            ok = ok and bool(reports) and failures == 0
        else:
            ok = ok and failures > 0 and reward['f2p_passed'] == 0
        summary['checks'][name] = {'expected_reward': expected, 'reward': reward,
                                   'passed': ok, 'junit': reports}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(name, json.dumps(summary['checks'][name]), flush=True)
    summary['passed'] = all(c['passed'] for c in summary['checks'].values())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not summary['passed']:
        raise AssertionError(f'Verifier controls failed: {output}/summary.json')
    print(f'PASS: all verifier controls passed: {output}', flush=True)


if __name__ == '__main__':
    main()
