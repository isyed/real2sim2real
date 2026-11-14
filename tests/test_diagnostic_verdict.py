from real2sim2real.controller_actor_training.diagnostic_verdict import DiagnosticVerdict


def test_diagnostic_warnings_do_not_become_control_acceptance():
    diagnostic = DiagnosticVerdict(continue_quality_warnings=True)
    assert not diagnostic.observe(1, warnings=["uncertain_marker"])
    assert not diagnostic.observe(2, warnings=["uncertain_marker"])
    result = diagnostic.finish(complete_horizon=True, task_completed=True)
    assert result["task_completed"] and not result["strict_quality_pass"]
    assert (
        not result["control_accepted"] and result["warnings"]["uncertain_marker"] == 1
    )
    assert diagnostic.observe(3, hard_failures=["physical_collision"])
    assert not diagnostic.finish(complete_horizon=True, task_completed=True)[
        "task_completed"
    ]
    assert DiagnosticVerdict().observe(1, warnings=["uncertain_marker"])
    assert not DiagnosticVerdict().finish(complete_horizon=False, task_completed=True)[
        "task_completed"
    ]
