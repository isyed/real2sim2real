import hashlib
import importlib.util
from pathlib import Path

import pytest

FILE = Path(__file__).parents[1] / 'modules/controller-actor-training/src/real2sim2real/controller_actor_training/policy_cache.py'
spec = importlib.util.spec_from_file_location('policy_cache', FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
KEY = '0123456789abcdef.npz'


def source(root, name, data):
    p = root / name / 'assets/.cache/convex' / KEY
    p.parent.mkdir(parents=True)
    p.write_bytes(data)
    return p.parents[3]


def test_same_key_different_policies_retain_exact_geometry(tmp_path):
    a = source(tmp_path, 'a', b'validated A hulls')
    b = source(tmp_path, 'b', b'validated B hulls')
    target = tmp_path / 'runtime'
    receipt = m.seed_policy_caches({'a': a, 'b': b}, target)
    assert (target / 'a/convex' / KEY).read_bytes() == b'validated A hulls'
    assert (target / 'b/convex' / KEY).read_bytes() == b'validated B hulls'
    assert receipt['policies']['a']['files'][KEY] == hashlib.sha256(b'validated A hulls').hexdigest()
    assert receipt['geometry_modified'] is False
    assert m.seed_policy_caches({'a': a, 'b': b}, target) == receipt


def test_conflict_inside_policy_still_fails_before_other_copies(tmp_path):
    a = source(tmp_path, 'a', b'A')
    b = source(tmp_path, 'b', b'B')
    target = tmp_path / 'runtime'
    dest = target / 'b/convex' / KEY
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'conflicting')
    with pytest.raises(ValueError, match='within b'):
        m.seed_policy_caches({'a': a, 'b': b}, target)
    assert not (target / 'a').exists()
    assert dest.read_bytes() == b'conflicting'


@pytest.mark.parametrize('name', ['../escape', '/absolute', '.', 'a/b'])
def test_invalid_namespace(tmp_path, name):
    with pytest.raises(ValueError, match='namespace'):
        m.seed_policy_caches({name: tmp_path / 'policy'}, tmp_path / 'runtime')


def test_symlink_is_not_followed(tmp_path):
    a = source(tmp_path, 'a', b'A')
    target = tmp_path / 'runtime'
    target.mkdir()
    (target / 'a').symlink_to(tmp_path / 'other', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        m.seed_policy_caches({'a': a}, target)


def test_receipt_empty_optional_policy_and_unrelated_global_cache(tmp_path):
    a = source(tmp_path, 'a', b'A')
    target = tmp_path / 'runtime'
    (target / 'convex').mkdir(parents=True)
    (target / 'convex' / KEY).write_bytes(b'old unscoped cache')
    proof = tmp_path / 'receipt.json'
    result = m.seed_policy_caches({'a': a, 'empty': tmp_path / 'empty'}, target, receipt=proof)
    assert result['policies']['empty']['files'] == {}
    assert proof.exists()
    assert (target / 'convex' / KEY).read_bytes() == b'old unscoped cache'
