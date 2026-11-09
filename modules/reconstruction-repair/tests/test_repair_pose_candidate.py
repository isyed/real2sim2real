import numpy as np
import pytest

from real2sim2real.artifacts import digest, load, publish
from real2sim2real.reconstruction_repair.pose_candidate import prepare


def sample(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    g = dict(frame_ids=np.arange(3), timestamps_s=np.arange(3) / 30,
        object_translation=np.zeros((3, 3)),
        object_rotation=np.tile(np.eye(3), (3, 1, 1)),
        vertices=np.eye(3), faces=np.array([[0, 1, 2]]),
        object_scale=np.array(1.0))
    np.savez(source / 'geometry.npz', **g)
    (source / 'quality.json').write_text('{"mean_iou":0.8}')
    publish(source, 'reconstruction', dict(geometry=source / 'geometry.npz',
        quality=source / 'quality.json'))
    return source, g


def proposal(tmp_path, source, g):
    path = tmp_path / 'proposal.npz'
    np.savez(path, **g)
    return path, dict(diagnostic_only=True, clean_motion_used=False,
        decision='Retain a source-only pose diagnostic for inspection',
        parent_artifact_sha256=digest(source / 'artifact.json'),
        geometry_sha256=digest(path))


def test_full_pose_candidate_preserves_parent_and_quality_scope(tmp_path):
    source, g = sample(tmp_path)
    g['object_translation'][1, 0] = .03
    g['object_rotation'][1] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    path, evidence = proposal(tmp_path, source, g)
    before = digest(source / 'artifact.json')
    report = prepare(source, path, tmp_path / 'out', evidence=evidence, hardlink=True)
    result = load(tmp_path / 'out')
    assert digest(source / 'artifact.json') == before
    saved = np.load(tmp_path / 'out' / result['files']['geometry']['path'])
    np.testing.assert_array_equal(saved['object_rotation'], g['object_rotation'])
    np.testing.assert_array_equal(saved['object_translation'], g['object_translation'])
    assert report['complete_timeline'] and report['source_frames'] == 3
    assert report['human_mesh_camera_timing_unchanged']
    assert not report['full_reference_approved']
    assert 'quality' not in result['files'] and 'parent_quality' in result['files']


@pytest.mark.parametrize('field', ['vertices', 'frame_ids', 'timestamps_s', 'object_scale'])
def test_rejects_non_pose_edits(tmp_path, field):
    source, g = sample(tmp_path)
    g[field].flat[0] += 1
    path, evidence = proposal(tmp_path, source, g)
    with pytest.raises(ValueError, match='immutable'):
        prepare(source, path, tmp_path / 'out', evidence=evidence)


@pytest.mark.parametrize('field', ['decision', 'clean_motion_used', 'parent_artifact_sha256', 'geometry_sha256'])
def test_candidate_requires_explicit_hash_bound_decision(tmp_path, field):
    source, g = sample(tmp_path)
    path, evidence = proposal(tmp_path, source, g)
    evidence.pop(field)
    with pytest.raises(ValueError, match='hash-bound candidate decision'):
        prepare(source, path, tmp_path / 'out', evidence=evidence)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('bad', ['reflection', 'zero_scale', 'nan'])
def test_rejects_invalid_pose(tmp_path, bad):
    source, g = sample(tmp_path)
    if bad == 'reflection':
        g['object_rotation'][1, 0, 0] = -1
    elif bad == 'zero_scale':
        g['object_scale'] = np.array(0.)
    else:
        g['object_translation'][1, 0] = np.nan
    path, evidence = proposal(tmp_path, source, g)
    with pytest.raises(ValueError):
        prepare(source, path, tmp_path / 'out', evidence=evidence)
