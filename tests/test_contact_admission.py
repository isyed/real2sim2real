import numpy as np
from real2sim2real.controller_actor_training.contact_admission import assess


def test_valid_foot_task_does_not_require_or_fabricate_hand_contact():
    phases=np.zeros((2,100));collision=dict(robot_object_collision=True,robot_scope='all')
    proof=dict(schema='real2sim2real.reference-contact-readiness/1',effectors=[dict(side='left_foot',frames_within_threshold=20)])
    result=assess(phases,collision,expected_effectors=('left_foot',),effector_audit=proof)
    assert result['controller_contact_compatible'] and result['zero_hand_phases_expected']
    assert not phases.any() and not result['physical_success_established']
    assert not assess(phases,collision)['controller_contact_compatible']


def test_nonhand_admission_requires_evidence_and_nonhand_collisions():
    phases=np.zeros((2,100));proof=dict(schema='real2sim2real.reference-contact-readiness/1',effectors=[dict(side='left_foot',frames_within_threshold=20)])
    assert not assess(phases,dict(robot_object_collision=True,robot_scope='hand'),expected_effectors=('left_foot',),effector_audit=proof)['controller_contact_compatible']
    assert not assess(phases,dict(robot_object_collision=True,robot_scope='all'),expected_effectors=('left_foot',))['controller_contact_compatible']
