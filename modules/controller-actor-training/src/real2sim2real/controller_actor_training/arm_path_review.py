"""Schematic arm-path failure review from saved IK, without contact dynamics."""

import argparse
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from real2sim2real.artifacts import digest, write_json
from .arm_path_audit import place_bases, hand_name


def review(path):
    import mujoco

    path = Path(path)
    audit = json.loads(path.read_text())
    for source, sha in audit["sources"].items():
        if digest(Path(source)) != sha:
            raise ValueError("Reviewed source changed")
    config = audit["config"]
    parent = json.loads(Path(config["trial"]).read_text())
    if digest(Path(config["trial"])) != config["sources"][config["trial"]]:
        raise ValueError("Trial changed")
    old = mujoco.MjModel.from_xml_path(
        str(Path(parent["configuration"]["output"]) / "scene/scene.xml")
    )
    model = mujoco.MjModel.from_xml_path(str(path.parent / "audit-scene.xml"))
    data = mujoco.MjData(model)
    states_path = Path(config["trial"]).with_name("states.npz")
    if digest(states_path) != parent["states_sha256"]:
        raise ValueError("States changed")
    with np.load(states_path) as f:
        states = f["qpos"].copy()
    rows = []
    pictures = []
    for number, row in enumerate(audit["placements"]):

        def penetration(s):
            return max(
                [
                    c["penetration_m"]
                    for c in s["arm_contacts"]
                    if not c.get("joint_neighbors", False)
                ],
                default=0,
            )

        sample = max(row["samples"], key=penetration)
        i = sample["source_index"]
        p = row["placement"]
        place_bases(model, config, p)
        data.qpos[:] = model.qpos0
        if "assembly" in config:
            for j in range(old.njnt):
                ident = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, old.joint(j).name
                )
                if ident >= 0 and old.jnt_type[j] in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    data.qpos[model.jnt_qposadr[ident]] = states[i, old.jnt_qposadr[j]]
        for a in config["arms"]:
            for name, q in zip(a["joints"], sample["arms"][a["name"]]["q"]):
                data.qpos[model.joint(name).qposadr[0]] = q
            names = (
                parent["candidate"]["joint_names"]
                if not a["source_prefix"]
                else parent["candidate"]["support"]["joint_names"]
            )
            for name in names:
                if "fixture_" not in name:
                    data.qpos[model.joint(hand_name(a, name)).qposadr[0]] = states[
                        i, old.joint(name).qposadr[0]
                    ]
        oq = old.joint("task_object_free").qposadr[0]
        nq = model.joint("task_object_free").qposadr[0]
        data.qpos[nq : nq + 7] = states[i, oq : oq + 7]
        mujoco.mj_fwdPosition(model, data)
        chains = {
            a["name"]: np.array(
                [
                    data.xpos[model.body(n).id]
                    for n in a["collision_bodies"] + [a["root"]]
                ]
            )
            for a in config["arms"]
        }
        gid = model.geom("table").id
        mid = model.geom_dataid[gid]
        va = model.mesh_vertadr[mid]
        table = (
            model.mesh_vert[va : va + model.mesh_vertnum[mid]]
            @ data.geom_xmat[gid].reshape(3, 3).T
            + data.geom_xpos[gid]
        )
        allpoints = np.concatenate([table, *chains.values()])
        center = (allpoints.min(0) + allpoints.max(0)) / 2
        scale = 420 / max(np.ptp(allpoints, axis=0).max(), 0.1)
        image = Image.new("RGB", (1000, 560), "#edf2f8")
        draw = ImageDraw.Draw(image)
        for column, axes in enumerate([(0, 1), (0, 2)]):

            def xy(p):
                return tuple(
                    (
                        (np.asarray(p) - center)[list(axes)] * [scale, -scale]
                        + [250 + 500 * column, 280]
                    ).tolist()
                )

            for point in table:
                draw.ellipse(
                    (*np.array(xy(point)) - 1, *np.array(xy(point)) + 1), fill="#c39159"
                )
            for side, points in chains.items():
                draw.line(
                    [xy(v) for v in points],
                    fill="#c34c4c" if side == "right" else "#286ec2",
                    width=7,
                )
                for point in points:
                    x, y = xy(point)
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="black")
            x, y = xy(data.xpos[model.body("task_object").id])
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#b54fc4")
            draw.text(
                (20 + 500 * column, 30),
                "Top XY" if column == 0 else "Side XZ",
                fill="black",
            )
        draw.text(
            (20, 8),
            f"{p['name']} | {sample.get('phase', 'sample')} | {sample['time_s']:.2f}s | fitted IK, NOT physics",
            fill="black",
        )
        draw.text(
            (20, 530),
            "Red: right arm. Blue: left arm. Brown: table vertices. Purple: object origin. Lines omit collision thickness.",
            fill="black",
        )
        picture = path.parent / f"arm-placement-{number}.jpg"
        image.save(picture, quality=92)
        pictures.append(dict(path=picture.name, sha256=digest(picture)))
        phases = []
        for phase in dict.fromkeys(s.get("phase", "unknown") for s in row["samples"]):
            ss = [s for s in row["samples"] if s.get("phase", "unknown") == phase]
            phases.append(
                dict(
                    phase=phase,
                    samples=len(ss),
                    arms={
                        a["name"]: dict(
                            position_mm=max(
                                s["arms"][a["name"]]["position_error_m"] for s in ss
                            )
                            * 1000,
                            rotation_deg=max(
                                s["arms"][a["name"]]["rotation_error_deg"] for s in ss
                            ),
                        )
                        for a in config["arms"]
                    },
                    maximum_arm_penetration_mm=max(penetration(s) for s in ss) * 1000,
                )
            )
        rows.append(
            dict(placement=p["name"], phases=phases, screen_pass=row["screen_pass"])
        )
    sections = []
    for row, picture in zip(rows, pictures):
        sections.append(
            "<h2>"
            + html.escape(row["placement"])
            + '</h2><img width="100%" src="'
            + picture["path"]
            + '"><table><tr><th>Phase</th><th>Wrist errors</th><th>Non-neighbor penetration mm</th></tr>'
            + "".join(
                "<tr><td>"
                + html.escape(r["phase"])
                + "</td><td>"
                + html.escape(str(r["arms"]))
                + "</td><td>"
                + str(round(r["maximum_arm_penetration_mm"], 2))
                + "</td></tr>"
                for r in row["phases"]
            )
            + "</table>"
        )
    page = path.parent / "review.html"
    page.write_text(
        '<!doctype html><meta charset="utf-8"><style>body{font:17px system-ui;max-width:1100px;margin:30px auto}td,th{border:1px solid #ccc;padding:8px}</style><h1>Arm feasibility and failed path intervals</h1><p>These are fitted kinematic skeletons, not physical rollouts. Declared robot base/table placement is assumed. Hand-body frames agree within the archived errors; collision meshes and dynamics are not certified equivalent. Adjacent joint housings remain in raw records but are excluded from non-neighbor collision metrics. Sampled velocity checks are lower bounds on unsampled peaks; no torque validation or continuous collision certification.</p><a href="audit.json">Complete audit</a>'
        + "".join(sections)
    )
    write_json(
        path.parent / "review-receipt.json",
        dict(
            audit_sha256=digest(path),
            implementation_sha256=digest(Path(__file__)),
            pictures=pictures,
            phases=rows,
            physics_rerun=False,
            rendered_scene_sha256=digest(path.parent / "audit-scene.xml"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if not a.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                "--audit",
                str(a.audit.resolve()),
                "--worker",
            ],
            timeout=120,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    review(a.audit)


if __name__ == "__main__":
    main()
