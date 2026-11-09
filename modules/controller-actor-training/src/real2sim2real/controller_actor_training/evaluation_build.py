"""One reproducible evaluator identity for frozen and residual policies.

Both policy branches belong to this build. Never rewrite provenance on saved
rollouts to make different producers appear identical.
"""
from pathlib import Path
import hashlib


def provenance(**kwargs):
    from flash_chord.evaluation.harness import evaluation_provenance
    value = evaluation_provenance(**kwargs)
    here = Path(__file__).parent
    build = hashlib.sha256(value['eval_code_sha256'].encode())
    # Include the complete public controller module, including both branches.
    for path in sorted(here.rglob('*.py')):
        build.update(str(path.relative_to(here)).encode())
        build.update(path.read_bytes())
    import flash_chord
    # Runtime actions, contacts, scene import and reward/observation configuration
    # are part of the producer too, not just the trajectory recorder.
    for root in sorted(map(Path, flash_chord.__path__)):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml') and '__pycache__' not in path.parts:
                build.update(str(path.relative_to(root)).encode())
                build.update(path.read_bytes())
    value['eval_code_sha256'] = build.hexdigest()
    value['eval_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return value
