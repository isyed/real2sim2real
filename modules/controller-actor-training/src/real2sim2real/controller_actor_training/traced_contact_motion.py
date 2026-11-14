"""Diagnose material sliding from saved integration inputs on separate MjData.

The caller supplies the original configured model. Force reproduction is
reported, not assumed: a changed model or solver may invalidate the audit.
No steps are executed and no source acceptance verdict is changed.
"""

import json
from pathlib import Path

import numpy as np

from .contact_patch_motion import summarize


def audit_trace(model, trace_directory, contact_audit, *, stride=10):
    """Re-solve a saved MujocoStepTrace and compare with its cached pad forces.

    `contact_audit` is a ContactPatchMotion instance for this model. Its settings
    govern sampling-gap integration; missing intervals are never bridged.
    The trace's after-step contacts belong to the pre-integration kinematics.
    """
    import mujoco

    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Positive integer sample stride required")
    root = Path(trace_directory)
    metadata = json.loads((root / "summary.json").read_text())
    trace = json.loads((root / "steps.json").read_text())
    signature = metadata["integration_signature"]
    if signature != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError("Complete integration inputs required")
    scratch = mujoco.MjData(model)
    objects = {
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == contact_audit.object_id
    }
    names = {model.geom(i).name: label for i, label in contact_audit.labels.items()}
    records, errors = [], []
    with np.load(root / "steps.npz", allow_pickle=False) as arrays:
        states = arrays["pre_integration"]
        times = arrays["before_time_s"]
        if (
            len(trace) != len(states)
            or len(times) != len(states)
            or not len(states)
            or states.shape != (len(states), mujoco.mj_stateSize(model, signature))
            or not np.isfinite(states).all()
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Aligned finite, chronological integration trace required")
        # Keep the terminal sample: contact loss can occur in the few steps
        # omitted by a regular stride, precisely where diagnosis matters most.
        indices = sorted(set(range(0, len(states), stride)) | {len(states) - 1})
        for i in indices:
            mujoco.mj_setState(model, scratch, states[i], signature)
            if not np.isclose(scratch.time, times[i], atol=1e-12, rtol=0):
                raise ValueError("Integration timestamps disagree")
            mujoco.mj_forward(model, scratch)
            sample = contact_audit.sample(model, scratch)
            cached = dict.fromkeys(names.values(), 0.0)
            for contact in trace[i]["after"]["contacts"]:
                a, b = contact["geoms"]
                pad = (
                    a
                    if a in names and b in objects
                    else b
                    if b in names and a in objects
                    else None
                )
                if pad is not None:
                    force = float(contact["wrench_contact_frame"][0])
                    if not np.isfinite(force):
                        raise ValueError("Finite cached contact forces required")
                    cached[names[pad]] += max(0.0, force)
            errors.extend(
                abs(sample["pads"][k]["force_n"] - v) for k, v in cached.items()
            )
            records.append(
                {
                    "time_s": float(scratch.time),
                    "phase": "trace",
                    "contact_patch_motion": sample,
                }
            )
    return {
        "schema": "real2sim2real.traced-contact-motion/1",
        "physics_steps": 0,
        "scratch_forward_only": True,
        "force_reconstruction_max_error_n": max(errors, default=None),
        "settings": contact_audit.settings,
        "sample_stride": stride,
        "summary": summarize(records, contact_audit.settings, phase="trace"),
        "records": records,
        "acceptance_claim": False,
    }
