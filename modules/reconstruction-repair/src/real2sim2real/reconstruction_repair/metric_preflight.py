"""Mandatory metric-size evidence before normal fine-pose/human repair.

This validates the CURRENT bundle. After resizing and fresh tracking, calibrate
that new bundle again: a previous small-mesh receipt cannot certify a new one.
Large scale hypotheses remain retained even when later processing is blocked.
"""
import json
from pathlib import Path

import numpy as np

from .bundle import arrays_read, bundle_read, file_sha256
from .metric_calibration import CalibrationConfig, diagnose


MAX_SCALE_RATIO = 1.10
FINE_STAGES = ('scale', 'semantic_orientation', 'rotation_spikes', 'pose_recovery', 'temporal_translation')


def fine_repair_requested(config):
    return config.human_repair or any(getattr(config, name) for name in FINE_STAGES)


def inspect_metric_calibration(bundle, directory):
    directory = Path(directory)
    report_path = directory / 'calibration.json'
    evidence_path = directory / 'evidence.npz'
    provenance_path = directory / 'evidence_provenance.json'
    if not all(p.is_file() and not p.is_symlink() for p in (report_path, evidence_path, provenance_path)):
        raise ValueError('Metric-size calibration requires retained report/evidence/provenance; rerun calibrate-object for this input bundle')
    saved = json.loads(report_path.read_text()); provenance = json.loads(provenance_path.read_text())
    evidence = arrays_read(evidence_path)
    if (saved.get('evidence_sha256') != file_sha256(evidence_path)
            or saved.get('evidence_provenance_sha256') != file_sha256(provenance_path)
            or provenance.get('evidence_sha256') != file_sha256(evidence_path)
            or saved.get('provenance') != provenance):
        raise ValueError('Metric-size calibration evidence/provenance hashes changed')
    # diagnose checks the current manifest, observations, split and floor evidence.
    try:
        config = CalibrationConfig(**saved['config'])
    except (KeyError, TypeError) as exc:
        raise ValueError('Invalid retained metric calibration configuration') from exc
    current = diagnose(bundle, evidence, provenance, config)
    if any(saved.get(key) != value for key, value in current.items()):
        raise ValueError('Metric-size calibration report differs from recomputed evidence')
    factor = current['scale_multiplier']
    passed = bool(current['candidate_ready'] and 1 / MAX_SCALE_RATIO <= factor <= MAX_SCALE_RATIO)
    status = ('passed_current_size' if passed else 'resize_and_retrack_required'
              if current['candidate_ready'] else 'calibration_rejected')
    return dict(stage='metric_size_preflight', status=status, passed=passed,
        current_mesh_size_verified=passed, scale_multiplier=factor,
        maximum_retained_size_ratio=MAX_SCALE_RATIO, calibration_candidate_ready=current['candidate_ready'],
        calibration_reasons=current['reasons'], report_sha256=file_sha256(report_path),
        evidence_sha256=file_sha256(evidence_path), provenance_sha256=file_sha256(provenance_path),
        input_manifest_sha256=file_sha256(bundle.root/'manifest.json'),
        selected_mesh_changed=False, requires_resize_and_fresh_tracking=status == 'resize_and_retrack_required',
        calibration_decision_preserved=True, approved_for_use=False)


def window_metric_preflight(data, bundle_path, calibration_path):
    if bundle_path is None or calibration_path is None:
        raise ValueError('Metric-size calibration is required before fine window repair; supply --metric-bundle and --metric-calibration')
    bundle = bundle_read(bundle_path)
    result = inspect_metric_calibration(bundle, calibration_path)
    if not result['passed']:
        raise ValueError('Fine window repair blocked by metric-size preflight: ' + result['status'])
    g = bundle.geometry
    expected = g['vertices'] * g['object_scale']
    if (data['vertices'].shape != expected.shape or not np.allclose(data['vertices'], expected, rtol=0, atol=1e-7)
            or not np.array_equal(data['faces'], g['faces'])):
        raise ValueError('Window mesh differs from the metric-checked bundle')
    ids = np.searchsorted(g['frame_ids'], data['frame_ids'])
    if (np.any(ids >= len(g['frame_ids'])) or not np.array_equal(g['frame_ids'][ids], data['frame_ids'])
            or not np.allclose(data['intrinsics'], g['intrinsics'][ids], rtol=0, atol=1e-7)):
        raise ValueError('Window frame/camera identities differ from the metric-checked bundle')
    return result
