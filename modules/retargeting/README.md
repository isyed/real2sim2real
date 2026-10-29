# Retargeting

This is the actual constrained IK retargeter, not a learned motion generator.
It converts native human MHR predictions to SOMA while retaining per-frame
identity, then uses Pink/Pinocchio tasks and robot joint limits to fit the
robot's body, hands and root to that human/object reference.

The first included robot configuration is G1. Supply its complete URDF/mesh
asset closure and SOMA/MHR models externally. `configs/g1` is a template;
`asset://...` paths resolve under the explicit `assets` directory. To add a
robot, supply the frame-alignment and IK mapping JSON files and a matching
robot URDF. Retargeting support does not by itself supply a trained body
controller for a different robot.

Input source time is preserved. The current exporter rejects nonuniform frame
times rather than inventing a playback speed. It reports the floor alignment
source; an explicit `support_plane` is preferred over its body-derived floor
fallback. An optional `support_asset` must already be transformed to the
final robot frame and contain no external USD references.

Output is a portable `motion_v1` Parquet with object mesh/URDF closure and a
hand–object proximity schedule. The default 2 cm contact threshold labels
reference proximity; it is not evidence that the simulated fingers touched
or held the object. Physics evaluation follows retargeting.
