"""Real compute-node FUSE/Singularity cache isolation check; no models or APIs."""
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, 'src/deepswe')
from runtime import cache_overlay, container, run

with tempfile.TemporaryDirectory(dir=os.environ['DEEPSWE_STORAGE_RESULTS'], prefix='cache-check-') as directory:
    root = Path(directory)
    lower = root / 'lower'
    lower.mkdir()
    (lower / 'existing').write_text('baseline')
    with cache_overlay(lower, root / 'overlay') as merged:
        code, out = run(container(Path(os.environ['DEEPSWE_TASK_SIF']), root / 'home',
            [f'{merged}:/cache']) + ['bash', '-c', 'test "$(cat /cache/existing)" = baseline && printf changed > /cache/existing && printf private > /cache/new'], timeout=180)
        assert (merged / 'existing').read_text() == 'changed'
    assert (lower / 'existing').read_text() == 'baseline'
    assert not (lower / 'new').exists()
print('PASS: fakeroot container cache writes leave shared cache unchanged', flush=True)
