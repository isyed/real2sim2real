"""Read-only bounded traces around individual MuJoCo integration steps.

Use in a single-threaded diagnostic process. No extra forward computation is
performed on the live data, and no physics/controller option is modified.
"""

from collections import deque
from pathlib import Path

import numpy as np

from real2sim2real.artifacts import write_json


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def audit_contact_geometry(model, data, *, tolerance_m=1e-5):
    """Inspect cached contacts for nonfinite or grossly misplaced points.

    This is a diagnostic, not a collision filter. For bounded rigid geoms,
    compare the contact midpoint against each geom's bounding sphere, allowing
    the full separation/penetration and inclusion margin. Unbounded planes and
    unsupported geom types receive finite/frame checks only. Never recompute or
    modify live dynamics here; call on position-consistent scratch data.
    """
    import mujoco

    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Positive finite contact tolerance required")
    bounded = {
        int(getattr(mujoco.mjtGeom, "mjGEOM_" + name))
        for name in ["SPHERE", "CAPSULE", "ELLIPSOID", "CYLINDER", "BOX", "MESH"]
    }
    anomalies = []
    for i, contact in enumerate(data.contact):
        ids = [int(contact.geom1), int(contact.geom2)]
        frame = contact.frame.reshape(3, 3)
        finite = bool(
            np.isfinite(np.r_[contact.pos, frame.ravel(), contact.dist]).all()
        )
        error = float(np.linalg.norm(frame @ frame.T - np.eye(3)))
        bounds = []
        for geom in ids:
            if geom < 0 or int(model.geom_type[geom]) not in bounded:
                continue
            radius = float(model.geom_rbound[geom])
            distance = float(np.linalg.norm(contact.pos - data.geom_xpos[geom]))
            allowance = (
                abs(float(contact.dist))
                + abs(float(contact.includemargin))
                + tolerance_m
            )
            bounds.append(
                {
                    "geom": model.geom(geom).name,
                    "center_distance_m": distance,
                    "bounding_radius_m": radius,
                    "excess_m": distance - radius - allowance,
                }
            )
        if not finite or error > 1e-5 or any(b["excess_m"] > 0 for b in bounds):
            anomalies.append(
                {
                    "contact_index": i,
                    "geoms": [model.geom(g).name if g >= 0 else None for g in ids],
                    "position_m": contact.pos.copy(),
                    "distance_m": float(contact.dist),
                    "finite": finite,
                    "frame_orthogonality_error": error,
                    "bounds": bounds,
                }
            )
    return _json_value(
        {
            "contact_count": len(data.contact),
            "anomalies": anomalies,
            "tolerance_m": tolerance_m,
            "contacts_modified": False,
        }
    )


class MujocoStepTrace:
    """Capture every step in a trailing window, freezing it at the first fault.

    Full integration inputs include solver warmstart. Derived force/contact
    fields are cached values, labeled by capture stage; reading them never calls
    mj_forward on live data. On a reset, a separate data instance recomputes the
    failing forward dynamics from the saved pre-step integration inputs.
    """

    def __init__(self, model, data, output, *, window_s=1.0, controller_snapshot=None):
        import mujoco

        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError("Positive finite trace window required")
        self.mj, self.model, self.data = mujoco, model, data
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.signature = mujoco.mjtState.mjSTATE_INTEGRATION
        self.size = mujoco.mj_stateSize(model, self.signature)
        self.rows = deque(maxlen=max(2, int(np.ceil(window_s / model.opt.timestep))))
        self.controller_snapshot = controller_snapshot
        self.steps = 0
        self.fault = None
        self.max_abs_qacc = 0.0
        self.original = None

    def integration_state(self):
        state = np.empty(self.size)
        self.mj.mj_getState(self.model, self.data, state, self.signature)
        return state

    def _capture(self, data):
        m, mj = self.model, self.mj
        contacts = []
        for i, contact in enumerate(data.contact):
            force = np.zeros(6)
            mj.mj_contactForce(m, data, i, force)
            contacts.append(
                {
                    "geoms": [
                        m.geom(int(contact.geom1)).name,
                        m.geom(int(contact.geom2)).name,
                    ],
                    "distance_m": float(contact.dist),
                    "position_m": contact.pos.copy(),
                    "frame": contact.frame.copy(),
                    "wrench_contact_frame": force,
                    "efc_address": int(contact.efc_address),
                }
            )
        return {
            "time_s": float(data.time),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "qacc": data.qacc.copy(),
            "ctrl": data.ctrl.copy(),
            "qacc_warmstart": data.qacc_warmstart.copy(),
            "actuator_force": data.actuator_force.copy(),
            "qfrc_actuator": data.qfrc_actuator.copy(),
            "qfrc_constraint": data.qfrc_constraint.copy(),
            "qfrc_bias": data.qfrc_bias.copy(),
            "qfrc_passive": data.qfrc_passive.copy(),
            "solver_niter": data.solver_niter.copy(),
            "warnings": [int(w.number) for w in data.warning],
            "contacts": contacts,
            "constraints": {
                "type": data.efc_type.copy(),
                "id": data.efc_id.copy(),
                "force": data.efc_force.copy(),
                "position": data.efc_pos.copy(),
                "velocity": data.efc_vel.copy(),
            },
        }

    def __enter__(self):
        self.original = self.mj.mj_step
        self.mj.mj_step = self._step
        return self

    def _step(self, model, data, *args, **kwargs):
        if model is not self.model or data is not self.data or self.fault:
            return self.original(model, data, *args, **kwargs)
        if args or kwargs.get("nstep", 1) != 1:
            raise ValueError("Trace requires individual mj_step calls")
        before = self._capture(data)
        integration = self.integration_state()
        controller = self.controller_snapshot() if self.controller_snapshot else None
        result = self.original(model, data, **kwargs)
        after = self._capture(data)
        self.steps += 1
        self.rows.append(
            {
                "index": self.steps,
                "integration": integration,
                "before": before,
                "after": after,
                "controller": controller,
            }
        )
        value = np.max(np.abs(after["qacc"]), initial=0)
        self.max_abs_qacc = max(self.max_abs_qacc, float(value))
        clock_ok = np.isclose(
            after["time_s"] - before["time_s"], model.opt.timestep, rtol=0, atol=1e-9
        )
        warned = any(a > b for a, b in zip(after["warnings"], before["warnings"]))
        finite = all(np.isfinite(after[k]).all() for k in ["qpos", "qvel", "qacc"])
        if not clock_ok or warned or not finite:
            self.fault = {
                "step": self.steps,
                "before_s": before["time_s"],
                "after_s": after["time_s"],
                "clock_ok": bool(clock_ok),
                "new_warning": warned,
                "finite": finite,
            }
            scratch = self.mj.MjData(model)
            self.mj.mj_setState(model, scratch, integration, self.signature)
            self.mj.mj_forward(model, scratch)
            recovered = self._capture(scratch)
            recovered["contact_geometry_audit"] = audit_contact_geometry(model, scratch)
            np.savez_compressed(
                self.output / "failure-forward.npz",
                integration=integration,
                **{k: v for k, v in recovered.items() if isinstance(v, np.ndarray)},
            )
            write_json(self.output / "failure-forward.json", _json_value(recovered))
            self.save()
        return result

    def save(self):
        if not self.rows:
            return
        rows = list(self.rows)
        arrays = {
            "pre_integration": np.stack([r["integration"] for r in rows]),
            "indices": np.array([r["index"] for r in rows]),
        }
        for stage in ["before", "after"]:
            arrays[stage + "_time_s"] = np.array([r[stage]["time_s"] for r in rows])
            for key, value in rows[0][stage].items():
                if isinstance(value, np.ndarray):
                    arrays[stage + "_" + key] = np.stack([r[stage][key] for r in rows])
        np.savez_compressed(self.output / "steps.npz", **arrays)
        details = [
            {
                "index": r["index"],
                "controller": r["controller"],
                **{
                    stage: {
                        k: v
                        for k, v in r[stage].items()
                        if not isinstance(v, np.ndarray)
                    }
                    for stage in ["before", "after"]
                },
            }
            for r in rows
        ]
        write_json(self.output / "steps.json", _json_value(details))
        write_json(
            self.output / "summary.json",
            _json_value(
                {
                    "recorded_steps": len(rows),
                    "total_steps": self.steps,
                    "timestep_s": float(self.model.opt.timestep),
                    "first_saved_time_s": rows[0]["before"]["time_s"],
                    "last_saved_time_s": rows[-1]["after"]["time_s"],
                    "fault": self.fault,
                    "max_abs_cached_qacc": self.max_abs_qacc,
                    "integration_signature": int(self.signature),
                    "joint_names": [
                        self.model.joint(j).name for j in range(self.model.njnt)
                    ],
                    "dof_joint_ids": self.model.dof_jntid,
                    "joint_qpos_addresses": self.model.jnt_qposadr,
                    "joint_ranges": self.model.jnt_range,
                    "actuator_names": [
                        self.model.actuator(a).name for a in range(self.model.nu)
                    ],
                    "actuator_force_ranges": self.model.actuator_forcerange,
                    "actuator_force_limited": self.model.actuator_forcelimited,
                    "live_forward_calls_added": 0,
                    "derived_fields": "Cached dynamics at each capture stage; not recomputed at post-integration qpos.",
                    "failure_forward": "Separate MjData recomputation from pre-step integration inputs; no physics step.",
                }
            ),
        )

    def __exit__(self, *error):
        self.mj.mj_step = self.original
        self.save()
