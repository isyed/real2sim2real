"""Contact-conditioned reference adapter; opt-in, separate from time profiles."""

import hashlib
import json

from .reference_refinement import ReferenceRefinementBackend


class ContactReferenceRefinementBackend(ReferenceRefinementBackend):
    def _parse(self, edits):
        from real2sim2real.controller_actor_training.contact_conditioned_posture import (
            ContactConditionedPosture,
        )

        if not isinstance(edits, dict) or set(edits) != {"profile", "contact_feedback"}:
            raise ValueError("profile and contact_feedback are required")
        s = self.session
        feedback = ContactConditionedPosture(
            edits["contact_feedback"],
            s.config,
            s.ref,
            s.model.jnt_range[s.hand["joints"]],
            self.bounds,
        )
        # Validate the static profile before changing the live session.
        fingerprint = super().edit_fingerprint(edits["profile"])
        return feedback, fingerprint

    def edit_fingerprint(self, edits):
        feedback, fingerprint = self._parse(edits)
        if feedback.gain == 0:
            return fingerprint
        encoded = json.dumps(edits["contact_feedback"], sort_keys=True, allow_nan=False)
        return hashlib.sha256((fingerprint + encoded).encode()).hexdigest()

    def run(self, edits: dict) -> dict:
        if self.session.started:
            raise RuntimeError("Reference refinement can execute only once; no reset")
        feedback, _ = self._parse(edits)
        self.session.contact_refinement = feedback
        try:
            return super().run(edits["profile"])
        finally:
            self.edits = edits
