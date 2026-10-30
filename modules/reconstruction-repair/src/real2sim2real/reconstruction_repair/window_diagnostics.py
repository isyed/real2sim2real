"""Separate inherited, introduced and immutable failures in pose-window reports.

This explanation never relaxes acceptance or promotes a rejected hypothesis.
An unchanged bad locked anchor makes a strict window pass impossible, even when
other frames become smoother. Consumers can retain that partial benefit while
directing the next experiment at the actual blocker.
"""
import math


def explain(report, *, locked_frame_ids=()):
    config = report['config']
    before, after = report['before'], report['after']
    ids = [r['frame'] for r in before]
    if len(ids) != len(set(ids)) or ids != [r['frame'] for r in after]:
        raise ValueError('Before/after window frames differ or repeat')
    locked = set(locked_frame_ids)
    if not locked.issubset(ids):
        raise ValueError('Locked frame outside measured window')

    def failures(row):
        required = ('iou', 'precision', 'recall', 'depth_points', 'support_penetration_m')
        if any(not math.isfinite(float(row[k])) for k in required):
            raise ValueError('Nonfinite measured window evidence')
        result = set()
        if (row['iou'] < config['min_iou'] or row['precision'] < config['min_precision']
                or row['recall'] < config['min_recall']):
            result.add('image_gate')
        error = row['depth_error_m']
        if error is not None and not math.isfinite(float(error)):
            raise ValueError('Nonfinite depth error')
        if (row['depth_points'] < config['min_depth_points'] or error is None
                or error > config['max_depth_error_m']):
            result.add('depth_gate')
        if row['support_penetration_m'] > config['max_support_penetration_m']:
            result.add('support_gate')
        if row['behind_camera']:
            result.add('camera_gate')
        return result

    rows = []
    for old, new in zip(before, after):
        a, b = failures(old), failures(new)
        regression = old['iou'] >= config['min_iou'] and new['iou'] < old['iou']-config['max_frame_iou_drop']
        introduced = b-a
        if regression:
            introduced.add('previous_good_image_regressed')
        rows.append(dict(frame=old['frame'], locked=old['frame'] in locked,
            inherited=sorted(a & b), introduced=sorted(introduced), resolved=sorted(a-b),
            original_failures=sorted(a), candidate_failures=sorted(b)))
    blockers = [dict(frame=r['frame'], reasons=r['original_failures'])
                for r in rows if r['locked'] and r['original_failures']]
    return dict(schema='real2sim2real.window-failure-explanation/1', frames=rows,
        locked_anchor_blockers=blockers,
        strict_pass_requires_anchor_or_input_revision=bool(blockers),
        acceptance_unchanged=True, candidate_accepted=bool(report['decision']['accepted']),
        scope='Window image/depth/support explanation only; metric, interaction and full-clip gates remain separate',
        recommendation=('Resolve measured input/anchor errors before repeating an optimizer that must preserve those anchors.'
                        if blockers else 'Inspect introduced and unresolved failures; partial motion benefit does not override them.'))
