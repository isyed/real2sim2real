"""Seed isolated runtime collision caches without merging policy geometry.

A mesh/parameter key identifies an input, not a unique convex decomposition.
Different validated policies can therefore pin different hulls under one key.
Each caller-supplied policy namespace retains its exact bytes. A conflict
*within* that namespace remains an error. This stdlib-only module can run in
bootstrap environments before the scientific Python environment is restored.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def seed_policy_caches(policies, target_root, *, receipt=None):
    """Copy each policy/assets/.cache/convex into target_root/namespace/convex.

    Namespace strings must be safe single path components. No existing cache
    is overwritten, and all conflicts are checked before any cache is copied.
    Policy assets and decompositions are never modified. The returned hashes
    attest to copying only, not to physical validity or task success.
    """
    target_root = Path(target_root)
    pending = []
    records = {}
    for namespace, policy in sorted(policies.items()):
        if not isinstance(namespace, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', namespace):
            raise ValueError('Unsafe policy cache namespace')
        source_root = Path(policy) / 'assets/.cache/convex'
        target = target_root / namespace / 'convex'
        if any(p.is_symlink() for p in (source_root, *source_root.parents, target, *target.parents)):
            raise ValueError('Policy collision caches cannot use symlinks')
        files = {}
        for source in sorted(source_root.glob('*.npz')):
            if (not re.fullmatch(r'[0-9a-f]{16}\.npz', source.name)
                    or not source.is_file() or source.is_symlink()):
                raise ValueError('Invalid pinned collision-cache file')
            expected = _digest(source)
            dest = target / source.name
            if dest.is_symlink() or (dest.exists() and (not dest.is_file() or _digest(dest) != expected)):
                raise ValueError(f'Conflicting pinned collision-cache key within {namespace}: {source.name}')
            files[source.name] = expected
            pending.append((source, dest, expected))
        records[namespace] = dict(runtime_cache_dir=str(target.parent), files=files)
    for source, dest, expected in pending:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            # Exclusive creation prevents a concurrent writer from being replaced.
            try:
                with dest.open('xb') as stream, source.open('rb') as src:
                    shutil.copyfileobj(src, stream, 8 * 1024 * 1024)
            except FileExistsError:
                pass
        if _digest(dest) != expected:
            raise ValueError('Policy collision-cache copy changed; preserve it for inspection')
    result = dict(schema='real2sim2real.policy-cache/1', policies=records,
                  namespaces_isolated=True, geometry_modified=False,
                  physical_equivalence_established=False,
                  cache_key_still_checked_by_runtime=True)
    if receipt is not None:
        receipt = Path(receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temp = receipt.with_name(receipt.name + f'.{os.getpid()}.pending')
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temp.replace(receipt)
    return result
