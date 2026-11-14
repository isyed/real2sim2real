"""Explicitly seeded refinement backend; excluded from from-scratch experiments."""

from .backend import FrozenReferenceBackend


class ReferenceRefinementBackend(FrozenReferenceBackend):
    def __init__(self, bundle, config):
        super().__init__(bundle, config)
        self.bounds = dict(config["refinement_bounds"])
        self.edits = None
        self.tools = {"controller.run_reference": self.run}

    def run(self, edits: dict) -> dict:
        """Execute once with a validated pressure/posture profile; {} preserves the seed."""
        from real2sim2real.controller_actor_training.reference_refinement import (
            ReferenceRefinement,
        )

        if self.session.started:
            raise RuntimeError("Reference refinement can execute only once; no reset")
        if not isinstance(edits, dict):
            raise TypeError("A refinement object is required")
        edits = edits or {}
        if edits:
            self.session.refinement = ReferenceRefinement(
                edits,
                self.session.config,
                self.session.ref,
                self.session.model.jnt_range[self.session.hand["joints"]],
                self.bounds,
            )
        self.edits = edits
        return self.session.run()

    def edit_fingerprint(self, edits: dict) -> str:
        """Hash validated effective commands on this frozen feedback clock."""
        import hashlib

        import numpy as np
        from real2sim2real.controller_actor_training.reference_refinement import (
            ReferenceRefinement,
        )

        if not isinstance(edits, dict):
            raise TypeError("A refinement object is required")
        session = self.session
        profile = (
            ReferenceRefinement(
                edits,
                session.config,
                session.ref,
                session.model.jnt_range[session.hand["joints"]],
                self.bounds,
            )
            if edits
            else None
        )
        rows = []
        zero = np.zeros(len(session.config["finger_joints"]))
        for t in session.ref["reference_times_s"]:
            offset, goals, gain = (
                profile.command(t, zero)
                if profile
                else (
                    zero,
                    session.config["pressure_targets_n"],
                    session.config["pressure"]["gain_m_per_n_s"],
                )
            )
            rows.append(np.r_[offset, goals, gain])
        values = np.asarray(rows, dtype="<f8")
        values[values == 0] = 0  # Canonicalize negative zero.
        return hashlib.sha256(values.tobytes()).hexdigest()

    def finish(self, output):
        import numpy as np
        from real2sim2real.artifacts import write_json

        if not self.session.started:
            # A malformed proposal is a rejected policy, not a missing-file
            # infrastructure failure. Save only the actual unexecuted state.
            physics = output / "physics"
            physics.mkdir(exist_ok=False)
            d = self.session.data
            np.savez_compressed(
                physics / "states.npz",
                qpos=[d.qpos.copy()],
                qvel=[d.qvel.copy()],
                commands=[d.ctrl.copy()],
                timestamps_s=[float(d.time)],
            )
            write_json(
                physics / "result.json",
                {"executed": False, "scope": "Initial state only; no simulation steps"},
            )
        result = super().finish(output)
        write_json(
            output / "reference-refinement.json",
            {"edits": self.edits, "bounds": self.bounds, "empty_edit": not self.edits},
        )
        return result
