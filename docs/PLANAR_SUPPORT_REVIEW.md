# Reviewing a reconstructed tabletop

Treat footprint, depth, initial object placement and physical stability as
separate checks. A small object-to-surface gap does not establish adequate
support beneath the object's centre of mass.

`real2sim2real.retargeting.planar_outline` provides optional numeric helpers:

- `mask_outline` extracts a convex perimeter from a reviewed boolean mask of a
  visible planar surface. Choose unobstructed frames when the object hides an
  edge in the first frame. Verify a static camera before transferring outlines.
- `fit_plane` estimates a plane from that surface's depth points, with explicit
  up-direction, coverage, residual and extent checks.
- `inspect_plane` checks the fixed plane on independent held-out points. It
  never refits or rescales the plane to make the check pass. Stable RGB edges
  can coexist with inconsistent predicted depth.
- `slab` intersects the reviewed image contour with an independently supported
  plane and uses separately observed thickness. It returns a proposal and
  evidence record, not an automatic replacement of the source surface.

Retain input hashes, source frame IDs, mask review, camera evidence, any depth
gain, fit/held-out split and rejected measurements. If later depth fails, keep
the improved RGB footprint separate from the rejected plane/translation. Do
not infer hidden thickness from a white top-surface mask or shift the table to
fit an object trajectory. A visible footprint correction may still fail physics.

`controller_actor_training.object_stability.run(..., support_usda=...)` tests
the saved support geometry without a robot or assistance. Its result includes
support geometry hash, shape count and measured support motion; the NPZ keeps
all body poses. This distinguishes a fixed but misconstructed tabletop from a
moving support body. CPU settling is a diagnostic with its recorded solver
settings, not a substitute for guarded robot rollouts or task-success metrics.

Validate the object pose, collision proxy, mass/inertia and near-contact region
as well. Translating an object horizontally does not by itself repair a tilted
or irregular collision base. Any pose proposal also needs source-image/depth
checks before adoption. These helpers are opt-in; controller defaults and
existing artifact selections remain unchanged.
