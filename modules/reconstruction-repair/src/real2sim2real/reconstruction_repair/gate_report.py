"""Explain individual gates without changing selection policy or approving RL.

Importance is categorical, not a weighted score. Unknown checks stay unknown.
Actor training may address controller execution errors; it does not certify or
repair input geometry, coordinates, data integrity or an impossible reference.
"""
from pathlib import Path

from .bundle import file_sha256, json_atomic


def importance(code):
    if any(word in code for word in ('immutable', 'camera_gate', 'near_plane', 'locked', 'anchor', 'no_mesh', 'contract', 'metric_size')):
        return 'required_validity', 'Resolve before normal repair/training; actor corrections cannot repair this input contract.'
    if any(word in code for word in ('support', 'depth_gate', 'motion_step', 'frame_rotation', 'frame_displacement', 'semantic', 'hand_surface', 'contact_evidence')):
        return 'reference_feasibility', 'Review magnitude and affected frames. Actor corrections may handle modest executable errors, not invalid support or impossible target motion.'
    return 'reference_quality', 'A quality threshold needs contextual review. Better execution from actor training does not prove the reconstruction is correct.'


def gate(code, stage, status, *, observed=None, threshold=None, frames=None, note=None):
    level, action = importance(code)
    names = dict(image_gate='Minimum image overlap', previous_good_image_regressed='Per-frame image preservation',
        depth_gate='Foreground depth agreement', support_gate='Support nonpenetration', camera_gate='Camera visibility',
        motion_step_gate='Motion step limits', no_temporal_benefit='Temporal improvement',
        locked_anchor_changed='Trusted anchor preservation', frame_rotation_bound='Rotation correction bound',
        frame_displacement_bound='Translation correction bound')
    return dict(id=code, stage=stage, status=status, importance=level,
                observed=observed, threshold=threshold, affected_frames=frames,
                explanation=note or names.get(code,code.replace('_', ' ')), recommended_action=action)


def explain(report):
    schema = report.get('schema'); rows = []; components = []
    metric = report.get('metric_size_preflight')
    if metric is not None:
        rows.append(gate('metric_size_current_mesh', 'metric_size', 'passed' if metric.get('passed') else 'not_verified',
            observed=metric.get('scale_multiplier'), threshold=metric.get('maximum_retained_size_ratio'),
            note=metric.get('status', 'Metric-size evidence unavailable')))
    elif schema not in ('r2s2r_engines.object-metric-calibration/1','r2s2r_engines.external-comparison-gates/1'):
        rows.append(gate('metric_size_current_mesh', 'metric_size', 'not_checked',
            note='This report does not contain a verified current-mesh metric preflight. Do not infer a pass.'))
    if schema == 'r2s2r_engines.repair-report/1':
        stages = list(report['stages']) + (report.get('human_repair') or {}).get('stages', [])
        for record in stages:
            name = record['stage']; outcome = record['outcome']
            first_gate = len(rows)
            components.append(dict(stage=name, outcome=outcome,
                selected=outcome == 'accepted', fallback='retained parent' if outcome != 'accepted' else None))
            validation = record.get('validation')
            if validation:
                checks = validation.get('checked_gates')
                if checks:
                    for check in checks:
                        rows.append(gate(check['id'], name, check['status'], note=check.get('explanation'),
                            observed=check.get('observed'),threshold=check.get('threshold'),frames=check.get('affected_frames')))
                else:
                    # Old reports do not enumerate passing subchecks. Do not invent them.
                    for reason in validation.get('reasons', []): rows.append(gate(reason, name, 'failed'))
                    rows.append(gate('proposal_selection', name, 'passed' if validation['accepted'] else 'failed',
                        note='Aggregate selection result; older report does not enumerate every passing subcheck.'))
            else:
                rows.append(gate('proposal_selection', name, outcome, note=record.get('reason')))
            for row in rows[first_gate:]:
                row['applies_to_selected_candidate'] = outcome == 'accepted'
        if report.get('aggregate_validation'):
            v = report['aggregate_validation']
            rows.append(gate('cumulative_proposal_selection', 'final_selection', 'passed' if v['accepted'] else 'failed',
                observed=v.get('reasons'), note='Final cumulative comparison; superseded proposals are not selected.'))
            rows[-1]['applies_to_selected_candidate'] = bool(v['accepted'])
    elif schema == 'r2s2r_engines.temporal-object-window-result/1':
        reasons = set(report['decision']['reasons']); c = report['config']; after = report['after']; before = report['before']
        checks = [
            ('image_gate', [r['frame'] for r in after if min(r['iou']/c['min_iou'],r['precision']/c['min_precision'],r['recall']/c['min_recall'])<1],
             dict(min_iou=min(r['iou'] for r in after)), dict(iou=c['min_iou'],precision=c['min_precision'],recall=c['min_recall'])),
            ('previous_good_image_regressed', [b['frame'] for a,b in zip(before,after) if a['iou']>=c['min_iou'] and b['iou']<a['iou']-c['max_frame_iou_drop']],
             dict(worst_iou_delta=min(b['iou']-a['iou'] for a,b in zip(before,after))), -c['max_frame_iou_drop']),
            ('depth_gate', [r['frame'] for r in after if r['depth_points']<c['min_depth_points'] or r['depth_error_m'] is None or r['depth_error_m']>c['max_depth_error_m']],
             dict(max_depth_error_m=max((r['depth_error_m'] for r in after if r['depth_error_m'] is not None),default=None)), c['max_depth_error_m']),
            ('support_gate', [r['frame'] for r in after if r['support_penetration_m']>c['max_support_penetration_m']],
             dict(max_penetration_m=max(r['support_penetration_m'] for r in after)), c['max_support_penetration_m']),
            ('camera_gate', [r['frame'] for r in after if r['behind_camera']], None, 'in front of camera'),
            ('motion_step_gate', None, report['decision']['motion_after'], dict(max_step_m=c['max_step_m'],max_rotation_step_deg=c['max_rotation_step_deg'])),
            ('frame_displacement_bound', None, None, c['maximum_frame_change_m']),
            ('frame_rotation_bound', None, None, c['rotation_bound_deg']),
            ('locked_anchor_changed', None, None, 'exact preservation'),
            ('no_temporal_benefit', None, dict(before=report['decision']['motion_before'],after=report['decision']['motion_after']), c['minimum_motion_improvement'])]
        for code, frames, observed, threshold in checks:
            rows.append(gate(code, 'temporal_window', 'failed' if code in reasons else 'passed',
                             frames=frames, observed=observed, threshold=threshold))
        components.append(dict(stage='temporal_window', outcome='accepted' if report['decision']['accepted'] else 'rejected',
            selected=report['decision']['accepted'], scope='This window proposal only; not the entire episode or its parent mesh.'))
    elif schema == 'r2s2r_engines.object-metric-calibration/1':
        rows.append(gate('metric_size_calibration_evidence', 'metric_size', 'passed' if report['candidate_ready'] else 'failed',
            observed=dict(scale_multiplier=report['scale_multiplier'],reasons=report['reasons']),
            note='Passing supports a size hypothesis; it does not certify the old mesh or old poses after resizing.'))
        rows.append(gate('metric_size_current_mesh', 'metric_size', 'not_verified',
            note='Calibrate the current resized/retracked input before normal fine repair; this is a coarse size hypothesis.'))
        components.append(dict(stage='calibration', outcome=report['status'], selected=False, scope='Mesh hypothesis requiring fresh tracking'))
    elif schema == 'r2s2r_engines.external-comparison-gates/1':
        allowed_status = {'passed','failed','not_checked','not_verified','review_required','disabled'}
        allowed_importance = {'required_validity','reference_feasibility','reference_quality','controller_execution','task_outcome'}
        for r in report['gates']:
            if r['status'] not in allowed_status or r['importance'] not in allowed_importance:
                raise ValueError('Invalid explicit comparison gate')
            rows.append(dict(r))
        components = report.get('components', [])
    else:
        raise ValueError('Unsupported gate-report source schema')
    if schema != 'r2s2r_engines.external-comparison-gates/1':
        rows += [dict(id='physics',stage='downstream',status='not_checked',importance='controller_execution',
                     explanation='This reconstruction report does not test robot physics.',recommended_action='Run an independent unassisted baseline.'),
                 dict(id='task_success',stage='downstream',status='not_checked',importance='task_outcome',
                     explanation='No task success is established by geometric checks.',recommended_action='Measure achieved motion and task metrics after physics.')]
    scope = report.get('comparison_scope', 'Reported repair proposal')
    if schema == 'r2s2r_engines.temporal-object-window-result/1':
        scope = f"Window proposal, frames {report['after'][0]['frame']}–{report['after'][-1]['frame']}; parent mesh is separate"
    elif schema == 'r2s2r_engines.object-metric-calibration/1':
        scope = 'Coarse mesh-size hypothesis; not a complete tracked reference'
    for row in rows:
        row.setdefault('candidate_scope',scope)
        row.setdefault('applies_to_selected_candidate',True)
        if type(row['applies_to_selected_candidate']) is not bool:
            raise ValueError('Gate selection scope must be an explicit boolean')
    blockers = [r['id'] for r in rows if r['applies_to_selected_candidate'] and r['importance']=='required_validity' and r['status'] not in ('passed','disabled')]
    feasibility = [r['id'] for r in rows if r['applies_to_selected_candidate'] and r['importance']=='reference_feasibility' and r['status'] not in ('passed','disabled')]
    return dict(schema='r2s2r_engines.comparison-gate-explanation/1', source_schema=schema, gates=rows,
        component_outcomes=components, required_checks_unresolved=blockers, reference_feasibility_review=feasibility,
        acceptance_policy_changed=False, weighted_score=None, actor_training_authorized=False,
        actor_training_recommendation=('Resolve required validity checks first.' if blockers else
            'Review reference feasibility before a bounded actor pilot; do not require perfect controller execution or task success first.' if feasibility else
            'A bounded actor pilot may be considered after scene/reference readiness and resource approval; this report does not authorize training.'),
        actor_can_potentially_improve=['balance','reach adjustments','contact/grasp timing','finger/body residual coordination'],
        actor_does_not_repair=['mesh size','coordinate/unit errors','missing supports','data integrity','incorrect or impossible target trajectories'])


def markdown(result):
    lines = ['# Comparison gates', '',
        'Each gate has its own scope and importance. A failed optional proposal does not discard a previously validated component. '
        'Rejected/superseded proposal gates remain visible but are not attributed to the retained parent. '
        'These explanations do not relax automatic acceptance, approve training or create a weighted score.', '',
        '| Candidate / stage / gate | Status | Importance | Observed / threshold | Explanation / next action |',
        '|---|---|---|---|---|']
    for r in result['gates']:
        facts = f"{r.get('observed')} / {r.get('threshold')}"
        detail = f"{r.get('explanation','')} {r.get('recommended_action','')}"
        if not r['applies_to_selected_candidate']: detail += ' Applies to an unselected proposal, not the retained output.'
        lines.append('| ' + ' | '.join(str(v).replace('|','/').replace('\n',' ') for v in
            [r.get('candidate_scope','')+' / '+r.get('stage','')+' / '+r['id'],r['status'],r['importance'],facts,detail]) + ' |')
    lines += ['', '## Actor-training interpretation', '', result['actor_training_recommendation'], '',
        'A trained residual may improve balance, reach, grasping and timing even when execution or task-success checks fail. '
        'It cannot certify wrong geometry or repair the target data. Small image-quality misses merit review; substantial '
        'support penetration or trajectory discontinuities need resolution first. No gate is silently overridden.', '']
    return '\n'.join(lines)


def write_explanation(report, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    result = explain(report)
    json_atomic(output/'gate_explanation.json', result)
    (output/'GATES.md').write_text(markdown(result))
    return result


def explain_file(source, output):
    import json
    source, output = Path(source), Path(output)
    if output.exists(): raise ValueError('Preserve existing comparisons; use a new output directory')
    report = json.loads(source.read_text()); result = write_explanation(report, output)
    json_atomic(output/'source.json',dict(path=str(source),sha256=file_sha256(source)))
    return result
