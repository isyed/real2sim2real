"""Fresh bilateral grasp hypotheses with common fitting, checks and fixture tests.

The strategy is data. It owns no physics, scene, acceptance or executable code.
Unknown contact observations remain unknown even if simulated mechanics pass.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import digest, write_json
from .paired_acquisition_audit import pinned_json
from .grasp_strategy_contract import validate_graph, workflow, PLAN_TOOL, HOLD_TOOL


def catalogue(config):
    from .object_grasp_pilot import load_scene
    from real2sim2real.retargeting.contact_layout import surface_layouts

    base = pinned_json(config["scene_config"])
    _, _, mesh = load_scene(base)
    settled_path = Path(base["output"]) / "settling.json"
    settled = json.loads(settled_path.read_text())
    mesh.apply_transform(settled["settled_world_from_object"])
    settings = base["planning"]
    diagnostics = {}
    layouts = surface_layouts(
        mesh,
        min_width_m=settings["min_width_m"],
        max_width_m=settings["max_width_m"],
        minimum_height_m=settings["minimum_height_m"],
        max_candidates=32,
        finger_offsets_m=settings["finger_offsets_m"],
        cues=[],
        yaw_samples=16,
        gravity_lever_weight=1.0,
        diagnostics=diagnostics,
    )
    result = {
        "schema": "real2sim2real.grasp-strategy-catalogue/1",
        "scene_config": config["scene_config"],
        "scene_receipt_sha256": digest(Path(base["output"]) / "scene/receipt.json"),
        "settling_sha256": digest(settled_path),
        "coordinate_frame": "gravity-settled task world; +Z up; metres",
        "object_bounds_m": mesh.bounds.tolist(),
        "center_of_mass_m": mesh.center_mass.tolist(),
        "center_of_mass_measured": False,
        "layouts": [dict(layout_id=i, **p) for i, p in enumerate(layouts)],
        "finger_offsets_m": settings["finger_offsets_m"],
        "diagnostics": diagnostics,
        "finite_contact_observations_verified": False,
        "control_ready": False,
    }
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "catalogue.json", result)
    return result


def deterministic_proposals(cat, count=4):
    """Rank geometry-only alternatives without using fitted or rollout outcomes."""
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError("One to eight deterministic strategies required")
    layouts = cat["layouts"]
    center = np.asarray(cat["center_of_mass_m"])
    ranked = []
    for i, first in enumerate(layouts):
        for second in layouts[i + 1 :]:
            a, b = np.array(first["center_m"]), np.array(second["center_m"])
            separation = np.linalg.norm(a - b)
            if separation < 0.035:
                continue
            # Prefer a contact pair straddling the assumed COM, without outcomes.
            score = np.linalg.norm((a + b) / 2 - center) - 0.1 * separation
            ranked.append((float(score), first["layout_id"], second["layout_id"]))
    ranked.sort()
    result = []
    for _, first, second in ranked[:count]:
        hands = {}
        for side, layout_id, sign in (("right", first, 1), ("left", second, -1)):
            hands[side] = dict(
                layout_id=layout_id,
                thumb_side=sign,
                wrist_roll_rad=0.0,
                opening_gap_m=0.008,
                preload_m=0.001,
                seed_flexion_rad=0.7,
                approach="above",
            )
        result.append(workflow(hands))
    return result


def fit_hand(base, mesh, settling, proposal, layout, side):
    import mujoco
    import trimesh
    from .object_grasp_pilot import load_scene, set_root_frame
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.contact_layout import contact_targets
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.pad_contacts import (
        pad_frame,
        observe_frames,
        fit_pad_contacts,
        contact_patch_samples,
    )

    offsets = list(base["planning"]["finger_offsets_m"])
    if proposal["thumb_side"] < 0:
        offsets.reverse()
    targets = contact_targets(mesh, layout, offsets, direction=proposal["thumb_side"])
    if targets is None:
        return {"accepted": False, "rejected_for": ["missing_local_opposing_surface"]}
    targets, gap, faces = trimesh.proximity.closest_point(mesh, targets)
    if gap.max() > 0.0005:
        return {"accepted": False, "rejected_for": ["target_projection_budget"]}
    outward = mesh.face_normals[faces]
    model, data, _ = load_scene(base)
    robots = [base["robot"], base["support_robot"]]
    robot = robots[side == "left"]
    prefix = robot.get("fixture_prefix", "fixture_")
    for other in robots:
        set_root_frame(
            model,
            data,
            other["root_body"],
            other.get("fixture_prefix", "fixture_"),
            [0.8, 0.8, 0.8],
            np.eye(3),
            preserve_state=True,
        )
    root = robot["root_body"]
    set_root_frame(model, data, root, prefix, [0, 0, 0], np.eye(3), preserve_state=True)
    idx = robot_indices(model, root)
    names = [model.joint(int(j)).name for j in idx["joints"]]
    seed = np.zeros(len(names))
    for i, name in enumerate(names):
        if name.endswith(("MCP_FE", "PIP", "DIP", "IP")):
            seed[i] = proposal["seed_flexion_rad"]
        if name.endswith("thumb_CMC_FE"):
            seed[i] = 1.2
    seed = np.clip(
        seed,
        model.jnt_range[idx["joints"], 0] + 1e-8,
        model.jnt_range[idx["joints"], 1] - 1e-8,
    )
    namespace = robot.get("namespace", "")
    frames = [
        pad_frame(
            model, root, namespace + side + "_" + finger + "_elastomer", [0, 1, 0]
        )
        for finger in ("thumb", "index", "middle")
    ]
    data.qpos[idx["q"]] = seed
    mujoco.mj_fwdPosition(model, data)
    local, _ = observe_frames(model, data, frames, idx)
    z = np.asarray(layout["close_axis"]) * proposal["thumb_side"]
    x = np.array([0.0, 0.0, -1.0])
    rotation = np.stack([x, np.cross(z, x), z], axis=1)
    rotation = (
        Rotation.from_rotvec(z * proposal["wrist_roll_rad"]).as_matrix() @ rotation
    )
    position = targets.mean(0) - rotation @ local.mean(0)
    set_root_frame(model, data, root, prefix, position, rotation, preserve_state=True)
    objq = model.joint("task_object_free").qposadr[0]
    data.qpos[objq : objq + 7] = settling["object_qpos"]
    kwargs = dict(max_nfev=base["planning"]["max_nfev"], fixture_prefix=prefix)
    fitted = fit_pad_contacts(
        model, data, root, frames, targets, -outward, seed, **kwargs
    )
    opened = fit_pad_contacts(
        model,
        data,
        root,
        frames,
        targets + proposal["opening_gap_m"] * outward,
        -outward,
        fitted["q"],
        **kwargs,
    )
    closed = fit_pad_contacts(
        model,
        data,
        root,
        frames,
        targets - proposal["preload_m"] * outward,
        -outward,
        fitted["q"],
        **kwargs,
    )
    pre = opened["q"].copy()
    direction = np.array([0, 0, 1.0]) if proposal["approach"] == "above" else -z
    xyz = [names.index(prefix + a) for a in "xyz"]
    pre[xyz] += base["rollout"]["approach_offset_m"] * direction
    path = path_audit(model, data, idx, [pre, opened["q"], closed["q"]])
    data.qpos[idx["q"]] = closed["q"]
    mujoco.mj_fwdPosition(model, data)
    patches = contact_patch_samples(model, data, frames, mesh)
    error = float(max(f["site_errors_m"].max() for f in (fitted, opened, closed)))
    normal = float(max(f["normal_errors_deg"].max() for f in (fitted, opened, closed)))
    reasons = []
    if error > base["planning"]["contact_error_m"]:
        reasons.append("pad_position_error")
    if normal > base["pad_planning"]["max_normal_error_deg"]:
        reasons.append("pad_normal_error")
    if path["max_penetration_m"] > base["planning"]["max_penetration_m"]:
        reasons.append("hand_path_penetration")
    if not path["within_joint_limits"]:
        reasons.append("hand_joint_limits")
    if not patches["all_pads_observed"]:
        reasons.append("missing_model_pad_patch")
    return dict(
        root_body=root,
        fixture_prefix=prefix,
        base_position_m=position.tolist(),
        base_rotation=rotation.tolist(),
        joint_names=names,
        q_pregrasp=pre.tolist(),
        q_open=opened["q"].tolist(),
        q_grasp=fitted["q"].tolist(),
        q_closed=closed["q"].tolist(),
        targets_m=targets.tolist(),
        outward_normals=outward.tolist(),
        site_errors_m=fitted["site_errors_m"].tolist(),
        maximum_target_error_m=error,
        maximum_normal_error_deg=normal,
        patches=patches,
        path=path,
        pad_plan={"frames": frames},
        reference_world_from_object=settling["settled_world_from_object"],
        nfev=sum(f["nfev"] for f in (fitted, opened, closed)),
        accepted=not reasons,
        rejected_for=reasons,
    )


def plan_pair(base, cat, hands):
    from .object_grasp_pilot import load_scene, set_root_frame
    from .pad_grasp_planning import path_audit
    from real2sim2real.retargeting.grasp_kinematics import robot_indices
    from real2sim2real.retargeting.paired_contacts import paired_wrench_screen

    settling = json.loads((Path(base["output"]) / "settling.json").read_text())
    _, _, mesh = load_scene(base)
    mesh.apply_transform(settling["settled_world_from_object"])
    layouts = {p["layout_id"]: p for p in cat["layouts"]}
    fitted = []
    for side in ("right", "left"):
        fitted.append(
            fit_hand(
                base,
                mesh,
                settling,
                hands[side],
                layouts[hands[side]["layout_id"]],
                side,
            )
        )
    reasons = [
        side + ":" + reason
        for side, hand in zip(("right", "left"), fitted)
        for reason in hand["rejected_for"]
    ]
    wrench = {"feasible": False, "reason": "missing_hand_geometry"}
    pair_path = None
    if all("patches" in h for h in fitted):
        wrench = paired_wrench_screen(
            fitted[0]["patches"],
            fitted[1]["patches"],
            mesh.center_mass,
            base["physics"]["mass_kg"],
            base["physics"]["friction"],
            base["pad_planning"]["max_normal_force_n"],
        )
        if not wrench["feasible"]:
            reasons.append("paired_static_gravity_balance")
        model, data, _ = load_scene(base)
        objq = model.joint("task_object_free").qposadr[0]
        data.qpos[objq : objq + 7] = settling["object_qpos"]
        for h in fitted:
            set_root_frame(
                model,
                data,
                h["root_body"],
                h["fixture_prefix"],
                h["base_position_m"],
                h["base_rotation"],
                preserve_state=True,
            )
        indices = [robot_indices(model, h["root_body"]) for h in fitted]
        combined = {
            k: np.concatenate([idx[k] for idx in indices])
            for k in ("q", "joints", "dofs", "geoms")
        }
        pair_path = path_audit(
            model,
            data,
            combined,
            [
                np.concatenate([h[k] for h in fitted])
                for k in ("q_pregrasp", "q_open", "q_closed")
            ],
        )
        if pair_path["max_penetration_m"] > base["planning"]["max_penetration_m"]:
            reasons.append("paired_path_collision")
        if not pair_path["within_joint_limits"]:
            reasons.append("paired_joint_limits")
    return dict(
        hands=fitted,
        parameters=hands,
        gravity_wrench=wrench,
        paired_path=pair_path,
        accepted=not reasons,
        rejected_for=reasons,
        settling=settling,
        contact_observation_admitted=False,
        physical_success_validated=False,
        control_ready=False,
    )


class StrategySession:
    def __init__(self, base, cat, parameters, output):
        self.base, self.cat, self.parameters, self.output = (
            base,
            cat,
            parameters,
            Path(output),
        )
        self.plan = self.hold = None

    def propose(self, hands: dict) -> dict:
        if self.plan is not None or hands != self.parameters:
            raise ValueError("Exactly one immutable admitted proposal required")
        self.plan = plan_pair(self.base, self.cat, hands)
        write_json(self.output / "plan.json", self.plan)
        return {
            "geometric_pass": self.plan["accepted"],
            "rejected_for": self.plan["rejected_for"],
        }

    def test_hold(self) -> dict:
        if self.plan is None or self.hold is not None:
            raise ValueError("One hold stage must follow its planned proposal")
        if not self.plan["accepted"]:
            self.hold = {
                "executed": False,
                "reason": "geometric_preflight_rejected",
                "simulated_hold_pass": False,
            }
        else:
            self.hold = run_hold(self.base, self.plan, self.output)
        write_json(self.output / "hold.json", self.hold)
        return self.hold


def run_hold(base, planned, output):
    from .object_grasp_pilot import trial
    from .grasp_feedback import relative_translation_drift

    settings = deepcopy(base)
    right, left = deepcopy(planned["hands"])
    candidate = dict(
        right, support=left, candidate_id=0, score=0.0, accepted=True, rejected_for=[]
    )
    planning = Path(output) / "planning"
    planning.mkdir()
    write_json(
        planning / "plan.json",
        {
            "scene_receipt_sha256": digest(Path(base["output"]) / "scene/receipt.json"),
            "settling": planned["settling"],
            "candidates": [candidate],
            "accepted": 1,
        },
    )
    settings["planning_directory"] = str(planning)
    result = trial(settings, candidate, Path(output) / "hold")
    end_close = sum(base["rollout"][k] for k in ("approach_s", "close_s"))
    rows = [r for r in result["records"] if r["time_s"] >= end_close]
    drifts = {}
    for role, prefix in (("primary", ""), ("support", "support_")):
        drifts[role] = (
            relative_translation_drift(
                [r["object_position_m"] for r in rows],
                [r[prefix + "wrist_position_m"] for r in rows],
                [r[prefix + "wrist_rotation"] for r in rows],
            )
            if rows
            else None
        )
    complete = (
        result["simulation_final_time_s"]
        >= result["simulation_requested_duration_s"] - settings["physics_step_s"]
    )
    passed = bool(
        result["success"]
        and complete
        and all(v is not None and v <= 0.01 for v in drifts.values())
    )
    return dict(
        executed=True,
        simulated_hold_pass=passed,
        original_success=result["success"],
        original_rejected_for=result["rejected_for"],
        acquisition_anchored_drift_m=drifts,
        complete=complete,
        control_ready=False,
        contact_observation_admitted=False,
        result_sha256=digest(Path(output) / "hold/result.json"),
    )


def evaluate(config):
    from .object_grasp_pilot import verify_scene_assets
    from gap.runtime.executor import WorkflowExecutor
    from gap_core.tools import ToolRegistry

    cat = pinned_json(config["catalogue"])
    base = pinned_json(cat["scene_config"])
    root, _ = verify_scene_assets(base)
    if (
        digest(root / "receipt.json") != cat["scene_receipt_sha256"]
        or digest(root.parent / "settling.json") != cat["settling_sha256"]
    ):
        raise ValueError("Scene or initial pose differs from the frozen catalogue")
    graph = pinned_json(config["workflow"])
    parameters = validate_graph(graph, {p["layout_id"] for p in cat["layouts"]})
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "workflow.json").write_bytes(Path(config["workflow"]["path"]).read_bytes())
    write_json(out / "protocol.json", config)
    snapshots = out / "implementation"
    snapshots.mkdir()
    for name in (
        "grasp_strategy.py",
        "grasp_strategy_contract.py",
        "multi_hand_scene.py",
        "pad_grasp_planning.py",
        "object_grasp_pilot.py",
    ):
        (snapshots / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    session = StrategySession(base, cat, parameters, out)
    registry = ToolRegistry()
    registry.register_callable(
        PLAN_TOOL,
        session.propose,
        summary="Fit and screen one paired grasp",
        requires_ctx=False,
    )
    registry.register_callable(
        HOLD_TOOL,
        session.test_hold,
        summary="Test a complete hold only after geometric admission",
        requires_ctx=False,
    )
    executor = WorkflowExecutor(
        out / "workflow.json",
        tool_registry=registry,
        trace_dir=out / "graph-trace",
        max_node_workers=1,
        node_visit_cap=3,
        checkpoints="off",
    )
    executor.execute()
    if executor.exit_status != "success" or session.hold is None:
        raise RuntimeError("Incomplete strategy evaluation")
    pinned_json(config["catalogue"])
    pinned_json(config["workflow"])
    pinned_json(cat["scene_config"])
    verify_scene_assets(base)
    report = dict(
        schema="real2sim2real.grasp-strategy-evaluation/1",
        config=config,
        graph_status=executor.exit_status,
        geometric_pass=session.plan["accepted"],
        rejected_for=session.plan["rejected_for"],
        hold=session.hold,
        control_ready=False,
        observed_contacts_verified=False,
        baseline_replaced=False,
    )
    write_json(out / "evaluation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "catalogue", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    deadline = config.get("deadline_s", 900)
    if type(deadline) not in (int, float) or not 0 < deadline <= 900:
        raise ValueError("Finite stage deadline up to 900s required")
    if not args.worker:
        from real2sim2real.bounded_process import run

        return run(
            [
                sys.executable,
                "-m",
                __spec__.name,
                args.operation,
                "--config",
                str(args.config.resolve()),
                "--worker",
            ],
            timeout=deadline,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    if args.operation == "prepare":
        from .multi_hand_scene import append_hypothesis_hand

        append_hypothesis_hand(config)
    else:
        {"catalogue": catalogue, "evaluate": evaluate}[args.operation](config)


if __name__ == "__main__":
    raise SystemExit(main())
