# Initial resting contact and clearance

A reconstructed object can match its image yet rock or fall when gravity is
enabled. Diagnose this before changing the controller or training a residual
actor. A successful isolated test does not establish successful manipulation.

1. Preserve the complete source reference and original physical trial. Screen
   the policy's actual imported collision shapes at its recorded reset pose
   using `controller_actor_training.policy_clearance.audit`. This respects the
   runtime's collision filters and primitive conversions. Contact witnesses are
   body-local; transform both witnesses before computing their signed gap.
   Missing generated contacts are not a global distance or clearance proof.
2. Run an isolated object/support settling test with the policy's initial pose,
   exact collision hulls, friction, damping, contact mode and solver settings.
   `policy_settling.run` also passes through declared `nconmax` and `njmax`:
   a larger diagnostic contact buffer is not silently called a matching policy
   test. Record the backend; a CPU result still requires GPU confirmation.
3. If RGB and geometry support a resting base, use an explicitly reviewed base
   axis and pivot with `reconstruction_repair.resting_pose.propose`. Bound the
   translation and rotation. Check mask overlap, predicted depth, support
   penetration and source-frame seams. A physically quieter pose that degrades
   the image beyond the declared preservation limit must remain rejected.
4. Test multiple-contact generation as an explicit solver ablation when a flat
   base rocks on sparse contact points. Use `solver_ablation.package`; do not
   change runtime defaults or assume this solves every object. Evaluate the
   pose and solver changes separately as well as together.
5. When only the first source pose needs repair, `initial_rest_reference.package`
   changes the four object pose representations at native frame zero. It
   requires one unarticulated rigid object, inactive initial hand-contact labels,
   consistent root/body representations and a bounded seam to frame one. It
   preserves robot targets, contact labels, geometry and every later pose.
   Rebuild the control-rate reference with `frozen.package` so interpolation and
   FK validation remain consistent. Preserve the reconstruction candidate and
   coordinate-transform evidence alongside the policy.
6. Repeat settling on the target GPU and then evaluate the complete frozen
   robot policy. Inspect object motion, robot/table clearance and measured
   contacts as well as task scores. Training requires the accepted scene and
   contact mode; `policy_physics.match_scene` rejects an implicit contact-mode
   change. An actor checkpoint from a different reference/physics setup must
   not be silently resumed.

These are opt-in diagnostic and packaging tools. They do not infer a valid
resting phase, reconstruct hidden motion, relax acceptance thresholds, approve
training or publish a competition submission.


`policy_clearance.mujoco_replay` provides an independent native MuJoCo CPU
contact-distance check at saved GPU joint/body poses. It verifies the policy
hash, body layout, joint layout and reproduced body positions before reporting
contacts. It performs no time integration and reports its sampling stride;
these are neither continuous-clearance nor measured-contact-force claims.
Use this cross-check before interpreting unusually large Newton triangle-mesh
witness gaps as physical penetration.

For a fresh actor, `policy_physics.match_contact_mode` explicitly adopts the
validated multiple-contact boolean in both configuration locations, then
`match_scene` checks the remaining solver settings and matches friction/damping.
The receipt records the previous mode and leaves resume compatibility unapproved.
Validate the staged worker entry point itself before cloud activation; a local
checkout can shadow staged orchestration during preparation.
