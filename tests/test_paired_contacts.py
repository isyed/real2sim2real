import copy
import json

import pytest

from real2sim2real.artifacts import digest
from real2sim2real.retargeting.paired_contacts import paired_wrench_screen
from real2sim2real.controller_actor_training.paired_contact_diagnostic import diagnose


def patch(sign):
    return dict(
        points_m=[[sign * 0.02, 0, 0]] * 2,
        normals=[[sign, 0, 0]] * 2,
        groups=[0, 0],
        samples_per_pad=[2],
    )


def test_pair_has_separate_pad_budgets_not_separate_sample_budgets():
    result = paired_wrench_screen(patch(-1), patch(1), [0, 0, 0], 0.15, 0.8, 1)
    assert result["feasible"] and result["pad_count"] == 2
    assert max(result["required_normal_forces_n"]) <= 1
    assert not result["robot_pair_validated"]
    assert not paired_wrench_screen(patch(-1), patch(1), [0, 0, 0], 0.2, 0.8, 1)[
        "feasible"
    ]
    missing = patch(1)
    missing["samples_per_pad"].append(0)
    assert not paired_wrench_screen(patch(-1), missing, [0, 0, 0], 0.15, 0.8, 1)[
        "feasible"
    ]
    corrupt = patch(1)
    corrupt["samples_per_pad"] = [3]
    with pytest.raises(ValueError):
        paired_wrench_screen(patch(-1), corrupt, [0, 0, 0], 0.15, 0.8, 1)


def test_paired_diagnostic_cannot_promote_unsupported_or_unpinned_inputs(tmp_path):
    candidates = []
    for sign in [-1, 1]:
        candidates.append(
            dict(
                candidate_id=sign,
                pair=dict(center_m=[sign * 0.02, 0, 0], center_of_mass_m=[0, 0, 0]),
                pad_plan=dict(
                    point_wrench_baseline=dict(
                        mass_kg=0.15, friction=0.8, max_normal_force_n=1
                    ),
                    sampled_patches=patch(sign),
                ),
            )
        )
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(candidates=candidates)))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            dict(
                config=dict(plan_sha256=digest(plan)),
                candidates=[
                    dict(candidate_id=i, contact_region_supported=False)
                    for i in [-1, 1]
                ],
            )
        )
    )
    config = dict(
        plan=str(plan),
        audit=str(audit),
        plan_sha256=digest(plan),
        audit_sha256=digest(audit),
        candidate_ids=[-1, 1],
        output=str(tmp_path / "out"),
    )
    original = plan.read_bytes()
    report = diagnose(config)
    row = report["pairs"][0]
    assert row["gravity_wrench"]["feasible"]
    assert not row["ready_for_robot_pair_fitting"] and not row["control_ready"]
    assert plan.read_bytes() == original
    bad = copy.deepcopy(config)
    bad["plan_sha256"] = "invalid"
    with pytest.raises(ValueError, match="Pinned"):
        diagnose(bad)
    mismatched = json.loads(audit.read_text())
    mismatched["config"]["plan_sha256"] = "different"
    audit.write_text(json.dumps(mismatched))
    config["audit_sha256"] = digest(audit)
    with pytest.raises(ValueError, match="different plan"):
        diagnose(config)
