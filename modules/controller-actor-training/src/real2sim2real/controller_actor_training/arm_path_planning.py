"""Bounded whole-path IK branch choices; never switch branches mid-trajectory."""


def initial_starts(arm, placement):
    starts = arm.get("initial_seeds_rad", [arm["seed_rad"]])
    choices = placement.get("ik_seed_indices", {})
    if arm["name"] not in choices:
        return starts
    index = choices[arm["name"]]
    if type(index) is not int or not 0 <= index < len(starts):
        raise ValueError("IK seed index must select a declared initial arm seed")
    return [starts[index]]


def rank_paths(rows):
    """Rank only complete passing sampled paths, keeping rejected paths visible.

    The audit owns all acceptance thresholds. Prefer lower speed utilization,
    then more joint travel margin; no pose error can outweigh a rejected path.
    This shortlist still needs dense screening and fresh loaded physics.
    """
    passing = [r for r in rows if r["screen_pass"] is True]
    return sorted(
        passing,
        key=lambda r: (
            max(a["maximum_sampled_velocity_ratio"] for a in r["arms"].values()),
            -min(a["minimum_joint_margin_rad"] for a in r["arms"].values()),
            r["placement"]["name"],
        ),
    )
