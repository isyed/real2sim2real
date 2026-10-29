import json
from pathlib import Path

import numpy as np
import pytest

from conftest import only
from real2sim2real.reconstruction_repair.bundle import arrays_atomic, file_sha256, json_atomic
from real2sim2real.reconstruction_repair.contracts import RepairProposal
from real2sim2real.reconstruction_repair.metric_calibration import calibrate_object
from real2sim2real.reconstruction_repair.metric_preflight import inspect_metric_calibration
from real2sim2real.reconstruction_repair.repairs import PROPOSERS
from real2sim2real.reconstruction_repair.runner import repair_run


def evidence(bundle, root, depth=2.):
    n = len(bundle.geometry['frame_ids']); root.mkdir()
    e = root / 'evidence.npz'; p = root / 'provenance.json'
    arrays_atomic(e, dict(frame_ids=bundle.geometry['frame_ids'],
        depth_m=np.full(bundle.observations['object_mask'].shape, depth), fit=np.arange(n) % 2 == 0,
        floor_planes=np.tile([0., -1., 0., 1.5], (n, 1)), floor_valid=np.ones(n, bool)))
    json_atomic(p, dict(schema='r2s2r_engines.metric-calibration-evidence/1', source='synthetic camera-Z depth',
        floor_roi_source='synthetic static floor', clean_trajectory_used=False,
        input_manifest_sha256=file_sha256(bundle.root / 'manifest.json'), evidence_sha256=file_sha256(e)))
    return e, p


def calibration(bundle, root, depth=2.):
    e, p = evidence(bundle, root, depth)
    target = root / 'calibration'; calibrate_object(bundle.root, e, p, target)
    return target


def test_default_blocks_before_any_fine_proposal_without_metric_evidence(make_bundle, tmp_path, monkeypatch):
    b = make_bundle(); called = []
    monkeypatch.setitem(PROPOSERS, 'pose_recovery', lambda *a: called.append(True))
    out = tmp_path / 'blocked'
    with pytest.raises(ValueError, match='Metric-size calibration is required'):
        repair_run(b.root, out)
    assert not called and not (out / 'checkpoints').exists() and not (out / 'bundle').exists()
    assert json.loads((out / 'status.json').read_text())['fine_repair_started'] is False


def test_default_accepts_plausible_current_size_and_locks_fine_scale(make_bundle, tmp_path):
    b = make_bundle(); checked = calibration(b, tmp_path / 'e')
    result = repair_run(b.root, tmp_path / 'fine', metric_calibration=checked)
    assert result.report['metric_size_verified']
    assert not result.report['diagnostic_only']
    assert next(s for s in result.report['stages'] if s['stage'] == 'scale')['outcome'] == 'disabled'
    assert file_sha256(result.output_bundle / 'geometry.npz') == file_sha256(b.root / 'geometry.npz')


def test_large_scale_hypothesis_is_retained_when_fine_repair_is_blocked(make_bundle, tmp_path):
    b = make_bundle(); e, p = evidence(b, tmp_path / 'e', depth=4.)
    out = tmp_path / 'fine'
    with pytest.raises(ValueError, match='resize_and_retrack_required'):
        repair_run(b.root, out, metric_evidence=e, metric_provenance=p)
    report = json.loads((out / 'metric_calibration/calibration.json').read_text())
    assert report['candidate_ready'] and report['scale_multiplier'] > 1.8
    assert (out / 'metric_calibration/calibrated_mesh.npz').exists()
    assert not (out / 'checkpoints').exists() and not (out / 'bundle').exists()


def test_calibration_cannot_certify_another_bundle_or_tampered_report(make_bundle, tmp_path):
    b = make_bundle(); checked = calibration(b, tmp_path / 'e')
    other = make_bundle(); other.manifest['sequence_id'] = 'other'
    json_atomic(other.root / 'manifest.json', other.manifest)
    with pytest.raises(ValueError, match='input hash'):
        inspect_metric_calibration(other, checked)
    report = json.loads((checked / 'calibration.json').read_text()); report['scale_multiplier'] = 1.09
    json_atomic(checked / 'calibration.json', report)
    with pytest.raises(ValueError, match='recomputed'):
        inspect_metric_calibration(b, checked)


def test_later_rejected_pose_does_not_discard_calibration(make_bundle, tmp_path, monkeypatch):
    b = make_bundle(); checked = calibration(b, tmp_path / 'e')
    before = file_sha256(checked / 'calibration.json')
    def bad_pose(bundle, current, config):
        g = {k: v.copy() for k, v in current.items()}; g['object_translation'][:, 0] += 1.
        return RepairProposal('pose_recovery', g, 'Deliberately rejected pose')
    monkeypatch.setitem(PROPOSERS, 'pose_recovery', bad_pose)
    result = repair_run(b.root, tmp_path / 'fine', only('pose_recovery'), metric_calibration=checked)
    assert result.report['metric_size_preflight']['passed']
    assert next(s for s in result.report['stages'] if s['stage'] == 'pose_recovery')['outcome'] == 'rejected'
    assert file_sha256(checked / 'calibration.json') == before


def test_auto_calibration_resume_keeps_stage_and_rejects_changed_evidence(make_bundle, tmp_path):
    b = make_bundle(); e, p = evidence(b, tmp_path / 'e'); out = tmp_path / 'fine'
    repair_run(b.root, out, metric_evidence=e, metric_provenance=p)
    stamp = (out / 'metric_calibration/calibration.json').stat().st_mtime_ns
    repair_run(b.root, out, metric_evidence=e, metric_provenance=p, resume=True)
    assert (out / 'metric_calibration/calibration.json').stat().st_mtime_ns == stamp
    p.write_text(p.read_text() + '\n')
    with pytest.raises(ValueError, match='evidence changed'):
        repair_run(b.root, out, metric_evidence=e, metric_provenance=p, resume=True)


def test_explicit_diagnostic_bypass_is_never_marked_metric_verified(make_bundle, tmp_path):
    b = make_bundle(); result = repair_run(b.root, tmp_path / 'diagnostic', diagnostic_only=True)
    assert result.report['diagnostic_only'] and not result.report['metric_size_verified']
    manifest = json.loads((result.output_bundle / 'manifest.json').read_text())
    assert manifest['repair']['diagnostic_only'] and not manifest['repair']['metric_size_verified']
    assert not result.report['approved_for_use']


def test_window_cli_boundary_also_requires_metric_size_evidence(tmp_path):
    from test_temporal_pose import fixture
    from real2sim2real.reconstruction_repair.temporal_pose import run_window
    source = tmp_path / 'window.npz'; arrays_atomic(source, fixture())
    provenance = tmp_path / 'source.json'
    json_atomic(provenance, dict(schema='r2s2r_engines.temporal-object-window/1', source='synthetic test',
                               clean_trajectory_used=False, input_sha256=file_sha256(source)))
    with pytest.raises(ValueError, match='Metric-size calibration is required'):
        run_window(source, provenance, tmp_path / 'window')
    assert not (tmp_path / 'window').exists()


def test_window_must_use_the_metric_checked_mesh_and_camera(make_bundle, tmp_path):
    from real2sim2real.reconstruction_repair.metric_preflight import window_metric_preflight
    b = make_bundle(); checked = calibration(b, tmp_path / 'e'); g = b.geometry
    data = dict(vertices=g['vertices'].copy(), faces=g['faces'].copy(),
                frame_ids=g['frame_ids'].copy(), intrinsics=g['intrinsics'].copy())
    assert window_metric_preflight(data, b.root, checked)['passed']
    data['vertices'] *= 2.5
    with pytest.raises(ValueError, match='mesh differs'):
        window_metric_preflight(data, b.root, checked)
    data['vertices'] = g['vertices'].copy(); data['intrinsics'][0, 0, 0] *= 2
    with pytest.raises(ValueError, match='camera identities'):
        window_metric_preflight(data, b.root, checked)


def test_normal_cli_fails_closed_without_metric_calibration(make_bundle, tmp_path):
    from real2sim2real.reconstruction_repair.cli import main
    b = make_bundle()
    assert main(['run', '--input', str(b.root), '--output', str(tmp_path / 'fine')]) == 2
    assert not (tmp_path / 'fine/bundle').exists()
