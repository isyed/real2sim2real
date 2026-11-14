# Loaded contact repair and early arm screening

Use a saved loaded response to distinguish wrist tracking error, object motion
relative to the wrist, pad-force deficits and actual clearance witnesses. A pad
fit at rest does not establish clearance after load transfer. The limiting link
can change after a repair; retain its geometry/body identity, not only a maximum
penetration scalar.

`retargeting.loaded_contact_repair.propose` preserves the observed loaded pad
positions and normals while adding selected-link collision costs. It operates
in scratch data with explicit bounds on every owned joint. The caller supplies
the current object and obstacle transforms. It returns a geometric proposal;
it does not preserve force support by construction. Replay acquisition and the
whole lift/hold with the original free object, actuators and acceptance checks.

For one failed snapshot:

1. Pin scene, source states, capture/controller clocks and command bounds.
2. Reproduce the original controller with zero posture change.
3. Fit bounded wrist/finger changes against the loaded contact locations. Keep
   full-hand collision costs and independent planning exclusions active.
4. Apply an explicit smooth transition; record both its timing and the pressure
   controller activation time. A force correction can preserve seating at rest
   yet activate too late to prevent relative drift under load.
5. Recheck the complete acquisition, pad-force gate and full hold. Earlier
   pressure activation, altered lift duration or larger command-offset bounds
   are controller changes and must be identified separately in comparisons.
   They must not change actuator force ceilings or acceptance limits.
6. Record failures and compare actual wrist/object motion. Reaching the nominal
   hold phase is not completion of the hold. Do not promote incomplete trials.

`controller_actor_training.arm_pose_screen.screen` tries a bounded set of IK
seeds for an explicit wrist target. It reports native joint margins and named
arm/body collision witnesses, retaining adjacent contacts separately. A passing
seed is preferred to a colliding seed with slightly smaller pose error. The live
state is preserved. All relevant arm and adapter collision bodies must be
declared by the caller.

Use it early to reject unsuitable base placements and rotation directions.
Label proposed inversion/placement poses as hypotheses when they are not
reconstructed targets. Independently passing endpoints may use different IK
branches. They do not establish a connecting path, velocity feasibility,
whole-hand clearance or an arm-mounted hold. Run the full path audit and loaded
arm checks before arm integration. Simulated mounting transforms and collision
proxies do not establish hardware adapter compatibility.

Keep contact-role evidence separate. An observed occupancy surface is not a
certified free-space model; a conservative exclusion is not physical material.
In particular, a closed convex container exterior cannot represent a rack prong
entering its opening or distinguish rim-only bearing from a fictitious cap.
Inspect only the support surfaces and approach corridor needed for the intended
placement, while retaining uncertainty elsewhere. See [control readiness](CONTROL_READINESS.md).

Tests cover bounded loaded proposals, live-state preservation, endpoint IK,
obstacle rejection and invalid transforms. Full replay equality and dynamics
outcomes remain downstream experiment evidence.
