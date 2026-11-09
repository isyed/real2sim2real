# Source-guided lower support-surface proposals

`reconstruction_repair.support_surface.project_below_support` offers an explicit
geometry hypothesis for an object observed resting on a plane. It moves only
mesh vertices below that plane along their original camera rays. The initial
pinhole projection stays unchanged; object poses and triangle topology stay
unchanged. This can isolate hidden lower-surface errors without globally
resizing foreground geometry that already agrees with predicted depth.

Supply a metric mesh, a rigid camera-frame pose, a unit oriented support plane,
reviewed resting evidence and explicit displacement, affected-fraction and
volume-change bounds. Search bounds do not replace source or physical admission
thresholds. Proposals exceeding bounds are reported rejected rather than
clipped to a passing value. Input arrays and untouched local vertices are
preserved. Triangle degeneracy and signed-volume changes are reported.

The default `projection_mode='camera_ray'` retains that projection-preserving
behavior. An explicit `projection_mode='support_normal'` instead moves each
affected vertex by the shortest distance to the plane. This can satisfy a
displacement bound when oblique camera rays would require a much larger edit,
but changes the initial outline. The receipt marks projection preservation
false and records its measured error. Both modes retain the same displacement,
affected-fraction, volume and degeneracy checks. Neither mode selects a mesh or
relaxes held-out image/depth/support acceptance.

This is a nonrigid mesh change. Projection invariance does not establish the
correct depth, wheel geometry, contact area or friction. A predicted floor is
also uncertain. The function does not infer a resting phase or choose a winner.
Validate held-out views and foreground depth, preserve inherited failed gates,
recompute mass properties, and run isolated CPU and target-GPU settling before
testing robot motion. Do not spread an initial-view claim to later occlusions
or turn physically relaxed poses into reference targets without source checks.

Different limitations can coexist: a successful object-support correction can
leave body tracking and handle reach unresolved. A frozen-controller failure
alone is not a prohibition on actor training, and a quiet object alone does not
establish a usable full-episode reference.

`reconstruction_repair.planar_support.propose` offers a separate, explicitly
reviewed contact-phase pose initializer. Its default fixes the object's tilt
and fits two bounded translations along the support plus rotation about its
normal. `allow_tilt=True` instead permits a bounded rotation around the observed
seed while preserving contact. Both retain the source mesh. Neither model is
appropriate for an unsupported lift, and neither is a trajectory or approval
mechanism. The receipt records the initial contact-adjusted pose, fitting limits
and the fact that the support is predicted. Validate actual triangles against
native masks and depth independently; a convex outline fit can fail those
checks even when its fitting objective improves. A rejected contact hypothesis
must not replace a source trajectory or be smoothed across missing evidence.
