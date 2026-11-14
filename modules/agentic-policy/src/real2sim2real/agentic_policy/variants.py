"""Create explicit, portable mass/friction/translation cases before a campaign."""

import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import write_json


def create_case(
    bundle,
    name,
    *,
    mass_scale=1.0,
    sliding_friction_scale=1.0,
    translation_m=(0.0, 0.0, 0.0),
):
    """Scale mass AND inertia; scale sliding friction on all scene surfaces.

    Scaling every compiled geom avoids a higher-friction partner masking a
    change to just the object under MuJoCo's default pair-friction combination.
    Torsional/rolling friction and all controller settings remain unchanged.
    """
    import mujoco

    bundle = Path(bundle).resolve()
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Simple case name required")
    shift = np.asarray(translation_m, float)
    if (
        shift.shape != (3,)
        or not np.isfinite(shift).all()
        or np.linalg.norm(shift) > 0.05
        or not 0.25 <= mass_scale <= 4
        or not 0.25 <= sliding_friction_scale <= 4
    ):
        raise ValueError("Finite bounded physical perturbation required")
    config = json.loads((bundle / "runtime.json").read_text())
    scene = (bundle / config["scene"]).resolve()
    if not scene.is_relative_to(bundle):
        raise ValueError("Portable scene required")
    model = mujoco.MjModel.from_xml_path(str(scene))
    obj = model.body(config["object_body"]).id
    joint = int(model.body_jntadr[obj])
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Free object required")
    tree = ET.parse(scene)
    body = next(b for b in tree.iter("body") if b.get("name") == config["object_body"])
    inertial = body.find("inertial")
    if inertial is None or "diaginertia" not in inertial.attrib:
        raise ValueError("Explicit body mass and diagonal inertia required")
    inertial.set("mass", repr(float(inertial.get("mass")) * mass_scale))
    inertial.set(
        "diaginertia",
        " ".join(
            repr(float(x) * mass_scale) for x in inertial.get("diaginertia").split()
        ),
    )
    world = tree.find("worldbody")
    for element in [world, *world.iter("body")]:
        if element is not world and not element.get("name"):
            raise ValueError("Named bodies required to map unnamed geoms unambiguously")
        bid = 0 if element is world else model.body(element.get("name")).id
        geoms = element.findall("geom")
        if len(geoms) != model.body_geomnum[bid]:
            raise ValueError("Flattened body geometry required for friction variants")
        for offset, geom in enumerate(geoms):
            gid = int(model.body_geomadr[bid]) + offset
            friction = model.geom_friction[gid].copy()
            friction[0] *= sliding_friction_scale
            geom.set("friction", " ".join(map(str, friction)))
    if list(tree.iter("pair")):
        raise ValueError(
            "Explicit pair overrides require a separately declared friction variant"
        )
    target_scene = scene.with_name(f"case-{name}.xml")
    if target_scene.exists():
        raise FileExistsError(target_scene)
    tree.write(target_scene, encoding="unicode")
    out = bundle / "cases" / name
    out.mkdir(parents=True, exist_ok=False)
    with np.load(bundle / config["initial"]) as a:
        q, v = a["qpos"].copy(), a["qvel"].copy()
    adr = model.jnt_qposadr[joint]
    q[adr : adr + 3] += shift
    np.savez(out / "initial.npz", qpos=q, qvel=v)
    c = deepcopy(config)
    c["scene"] = str(target_scene.relative_to(bundle))
    c["initial"] = str((out / "initial.npz").relative_to(bundle))
    write_json(out / "runtime.json", c)
    changed = mujoco.MjModel.from_xml_path(str(target_scene))
    assert np.isclose(changed.body_mass[obj], model.body_mass[obj] * mass_scale)
    assert np.allclose(changed.body_inertia[obj], model.body_inertia[obj] * mass_scale)
    assert np.allclose(
        changed.geom_friction[:, 0], model.geom_friction[:, 0] * sliding_friction_scale
    )
    assert np.array_equal(changed.geom_friction[:, 1:], model.geom_friction[:, 1:])
    report = {
        "name": name,
        "mass_scale": mass_scale,
        "object_mass_kg": float(changed.body_mass[obj]),
        "sliding_friction_scale_all_geoms": sliding_friction_scale,
        "translation_m": shift.tolist(),
        "backend": {
            "kind": "continuous",
            "runtime": str((out / "runtime.json").relative_to(bundle)),
        },
    }
    write_json(out / "case.json", report)
    return report
