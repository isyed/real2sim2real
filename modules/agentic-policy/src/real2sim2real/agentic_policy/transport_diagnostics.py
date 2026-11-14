"""Explain the enforced transport rejection without mistaking a logged pad overlap for it."""


def rejection_evidence(plan):
    """Return decision provenance from a rejected plan; never execute or relax it.

    Contact-aware admission supersedes the legacy stationary/carried screen
    conjunction. Its loaded-preview decision can in turn supersede geometric
    pad-only overlap. These are forecasts, not observed live collisions.
    """
    if plan.get("admitted") is not False:
        raise ValueError("An explicitly rejected transport plan is required")
    aware = plan.get("contact_aware_screen")
    if aware is not None:
        if aware.get("admitted") is not False:
            raise ValueError("Inconsistent contact-aware transport decision")
        preview = aware.get("dynamic_preview")
        if preview is not None and preview.get("passed") is False:
            blocker = {
                "source": "contact_aware_screen.dynamic_preview",
                "reason": preview.get("reason"),
                "peak_sampled_penetration_m": preview.get("peak_sampled_penetration_m"),
            }
        else:
            blocker = {
                "source": "contact_aware_screen",
                "reason": aware.get("reason", "contact-aware geometry rejected"),
                "peak_penetration_m": aware.get("peak_penetration_m"),
                "witness": aware.get("witness"),
                "assumption": aware.get("assumption"),
                "groups": aware.get("audit", {}).get("groups", {}),
            }
        blockers = [blocker]
    else:
        blockers = [
            {
                "source": name,
                **{
                    k: screen.get(k)
                    for k in ("reason", "peak_penetration_m", "witness", "assumption")
                },
            }
            for name in ("screen", "carried_object_screen")
            if (screen := plan.get(name)) is not None
            and screen.get("admitted") is False
        ]
        if not blockers:
            blockers = [
                {
                    "source": "plan",
                    "reason": plan.get("reason", "unspecified rejection"),
                }
            ]
    return {
        "admitted": False,
        "blockers": blockers,
        "live_collision_claim": False,
        "interpretation": "Enforced plan/forecast rejection; logged auxiliary overlaps are not necessarily the admission blocker.",
    }
