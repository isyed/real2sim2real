import numpy as np
import pytest

from real2sim2real.egocentric.tracking import refine_with_retries, refinement_schedule


def fake_tracker(values):
    state = dict(pose=np.array([10.0]), seeds=[])

    def restore(pose):
        state["pose"] = pose.copy()

    def propose(iterations):
        state["seeds"].append(state["pose"].copy())
        state["pose"] = np.array([values[iterations]], dtype=float)
        return state["pose"]

    return state, restore, propose


def test_retries_restart_from_accepted_seed_and_retain_rejected_candidate():
    state, restore, propose = fake_tracker({5: 100, 1: 80, 2: 11})
    pose, quality, attempts = refine_with_retries(
        np.array([10.0]),
        propose,
        lambda p: dict(accepted=p[0] < 20),
        restore,
        retries=[1, 2],
    )
    assert quality["accepted"] and pose[0] == 11 and state["pose"][0] == 11
    assert all(seed[0] == 10 for seed in state["seeds"])
    assert [a["pose"][0] for a in attempts] == [100, 80, 11]


def test_all_failures_restore_tracker_without_fabricating_acceptance():
    state, restore, propose = fake_tracker({5: 100, 1: 80})
    _, quality, attempts = refine_with_retries(
        np.array([10.0]),
        propose,
        lambda p: dict(accepted=False),
        restore,
        retries=[1],
    )
    assert not quality["accepted"] and state["pose"][0] == 10 and len(attempts) == 2


def test_passing_primary_does_not_run_extra_hypotheses():
    state, restore, propose = fake_tracker({5: 11})
    _, quality, attempts = refine_with_retries(
        np.array([10.0]),
        propose,
        lambda p: dict(accepted=True),
        restore,
        retries=[1],
    )
    assert quality["accepted"] and len(attempts) == 1 and len(state["seeds"]) == 1


def test_assessment_exception_restores_previous_state():
    state, restore, propose = fake_tracker({5: 100})

    def assess(_):
        raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError):
        refine_with_retries(np.array([10.0]), propose, assess, restore)
    assert state["pose"][0] == 10


@pytest.mark.parametrize(
    "primary,retries",
    [(0, []), (5, [5]), (5, [1, 2, 3, 4]), (5, [True]), (5, [1.5]), (11, [])],
)
def test_invalid_schedules_fail_before_inference(primary, retries):
    with pytest.raises(ValueError):
        refinement_schedule(primary, retries)
