"""Kinematic-only full-hand scenes for explicit hypothetical geometry checks."""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from real2sim2real.artifacts import digest


def prepare(robot_xml, root_name, meshes, output, *, spheres=(), rings=()):
    """Preserve all robot collision links and add a bounded six-joint wrist.

    Objects are fixed for IK/collision checks. The result must never be used as
    evidence of loaded retention: no free-object physics scene is produced.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve the existing conditional scene")
    source = Path(robot_xml).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None or root.find("worldbody") is None or root.find("asset") is None:
        raise ValueError("Explicit compiler, asset and worldbody sections required")
    meshdir = (source.parent / compiler.get("meshdir", "")).resolve()
    compiler.set("meshdir", str(meshdir))
    world = root.find("worldbody")
    hand = world.find("body")
    if hand is None or hand.get("name") != root_name or root.findall(".//freejoint"):
        raise ValueError("Declared fixed-root articulated hand required")
    if root.find("keyframe") is not None:
        raise ValueError("Explicit keyframe adaptation required")
    sources = {str(source): digest(source)}
    for mesh in root.findall("asset/mesh"):
        p = meshdir / mesh.get("file")
        sources[str(p)] = digest(p)
    for axis, vector in zip("xyz", np.eye(3)):
        ET.SubElement(
            hand,
            "joint",
            name="fixture_" + axis,
            type="slide",
            axis=" ".join(map(str, vector)),
            range="-1 1",
            damping="5",
        )
        ET.SubElement(
            hand,
            "joint",
            name="fixture_r" + axis,
            type="hinge",
            axis=" ".join(map(str, vector)),
            range="-.5 .5",
            damping=".1",
        )
    asset = root.find("asset")
    for name, path in meshes.items():
        path = Path(path).resolve()
        sources[str(path)] = digest(path)
        ET.SubElement(asset, "mesh", name=name, file=str(path))
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            rgba=".5 .5 .5 1",
            contype="1",
            conaffinity="1",
        )
    for name, position, radius in spheres:
        position = np.asarray(position, float)
        if (
            position.shape != (3,)
            or not np.isfinite(position).all()
            or not 0 < radius < 0.2
        ):
            raise ValueError("Bounded finite sphere required")
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="sphere",
            pos=" ".join(map(str, position)),
            size=str(radius),
            rgba=".9 .3 .2 .4",
            contype="1",
            conaffinity="1",
        )
    for name, center, radius, zmin, zmax in rings:
        center = np.asarray(center, float)
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite([radius, zmin, zmax]).all()
            or not (0 < radius < 0.2 and zmax > zmin)
        ):
            raise ValueError("Bounded ring exclusion required")
        # Solid cylindrical exclusion deliberately forbids hand entry into the rim.
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="cylinder",
            pos=f"{center[0]} {center[1]} {(zmin + zmax) / 2}",
            size=f"{radius} {(zmax - zmin) / 2}",
            rgba=".9 .5 .1 .3",
            contype="1",
            conaffinity="1",
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output)
    return dict(
        scene_sha256=digest(output),
        sources=sources,
        scope="conditional kinematics only; static object; no physics admission",
    )


def sample_paths(
    model, data, indices, waypoints, *, max_step_rad=0.02, max_step_m=0.002
):
    """Full owned-hand collision witnesses, plus palm/proximal classification.

    Uses all enabled collision geometry. Sampled joint interpolation is not a
    continuous collision certificate; no simulation steps are executed.
    """
    import mujoco

    q = np.asarray(waypoints, float)
    bounds = model.jnt_range[indices["joints"]]
    if (
        q.ndim != 2
        or q.shape[1] != len(bounds)
        or len(q) < 2
        or not np.isfinite(q).all()
        or not 0 < max_step_rad <= 0.02
        or not 0 < max_step_m <= 0.002
    ):
        raise ValueError("Finite bounded path required")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    owned = set(map(int, indices["geoms"]))
    step = np.where(
        model.jnt_type[indices["joints"]] == mujoco.mjtJoint.mjJNT_SLIDE,
        max_step_m,
        max_step_rad,
    )
    categories = {
        name: dict(penetration_m=0.0, witness=None)
        for name in ["table", "rack", "cup_pad", "cup_other", "protrusion", "self"]
    }
    samples = []
    for segment, (a, b) in enumerate(zip(q[:-1], q[1:])):
        count = max(2, int(np.ceil(np.max(abs(b - a) / step))) + 1)
        if count > 512:
            raise ValueError("Path sampling budget exceeded")
        for fraction, value in enumerate(np.linspace(a, b, count)):
            scratch.qpos[indices["q"]] = value
            mujoco.mj_fwdPosition(model, scratch)
            samples.append(value.copy())
            for c in scratch.contact:
                g1, g2 = int(c.geom1), int(c.geom2)
                if c.dist >= 0 or not ({g1, g2} & owned):
                    continue
                hand = g1 if g1 in owned else g2
                other = g2 if hand == g1 else g1
                hn = model.geom(hand).name
                on = model.geom(other).name
                kind = (
                    "self"
                    if other in owned
                    else "table"
                    if on == "table"
                    else "rack"
                    if on == "rack"
                    else "cup_pad"
                    if on == "cup" and hn.endswith("_elastomer")
                    else "cup_other"
                    if on == "cup"
                    else "protrusion"
                )
                if -c.dist > categories[kind]["penetration_m"]:
                    categories[kind] = dict(
                        penetration_m=float(-c.dist),
                        witness=dict(
                            segment=segment,
                            fraction=fraction / (count - 1),
                            geoms=[hn, on],
                            hand_body=model.body(model.geom_bodyid[hand]).name,
                            q=value.tolist(),
                        ),
                    )
    return dict(
        samples=len(samples),
        within_joint_limits=bool(((q >= bounds[:, 0]) & (q <= bounds[:, 1])).all()),
        categories=categories,
        maximum_penetration_m=max(v["penetration_m"] for v in categories.values()),
        continuous_collision_certified=False,
        full_hand_collision_geoms=len(owned),
    ), np.asarray(samples)


def align_pad_seed(points, normals, targets, target_normals, *, normal_length_m=0.03):
    """Proper rigid seed from actual pad locations and normals; no joint solve."""
    from scipy.spatial.transform import Rotation

    p, n, t, tn = [
        np.asarray(x, float) for x in (points, normals, targets, target_normals)
    ]
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or len(p) < 3
        or any(x.shape != p.shape or not np.isfinite(x).all() for x in (n, t, tn))
        or not np.isfinite(p).all()
        or not 0 < normal_length_m < 0.1
    ):
        raise ValueError(
            "At least three finite pad frames and bounded normal weight required"
        )
    source = np.r_[p - p.mean(0), n * normal_length_m]
    goal = np.r_[t - t.mean(0), tn * normal_length_m]
    rotation, _ = Rotation.align_vectors(goal, source)
    r = rotation.as_matrix()
    return t.mean(0) - r @ p.mean(0), r


def assess_obstacle_restoration(
    isolated,
    restored,
    *,
    required_scenarios,
    target_error_m,
    normal_error_deg,
    max_target_error_m,
    max_normal_error_deg,
    max_penetration_m,
):
    """Fail closed until the isolated fit AND every restored path pass.

    Caller must supply unchanged-path audits and verify complete scene restoration.
    This gate aggregates sampled kinematics only; it never admits physics or
    observed geometry. Missing or nonfinite measurements cannot become a pass.
    """
    required = tuple(required_scenarios)
    if not required or len(set(required)) != len(required):
        raise ValueError("Distinct required restoration scenarios needed")
    limits = np.asarray(
        [max_target_error_m, max_normal_error_deg, max_penetration_m], float
    )
    if not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Finite positive frozen acceptance limits required")
    errors = np.asarray([target_error_m, normal_error_deg], float)
    alignment = bool(
        np.isfinite(errors).all()
        and (errors >= 0).all()
        and (errors <= limits[:2]).all()
    )

    def path_pass(audit):
        penetration = audit.get("maximum_penetration_m", float("nan"))
        return bool(
            np.isfinite(penetration)
            and 0 <= penetration <= max_penetration_m
            and audit.get("within_joint_limits") is True
        )

    isolated_pass = alignment and path_pass(isolated)
    missing = [name for name in required if name not in restored]
    passed = {name: path_pass(restored[name]) for name in required if name in restored}
    return dict(
        alignment_pass=alignment,
        isolated_pass=isolated_pass,
        missing_scenarios=missing,
        restored_path_pass=passed,
        conditional_approach_admitted=bool(
            isolated_pass and not missing and all(passed.values())
        ),
        physics_admitted=False,
        observation_admitted=False,
        continuous_collision_certified=False,
    )
