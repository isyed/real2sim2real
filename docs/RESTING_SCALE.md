# Resting-phase camera-ray scale hypotheses

`reconstruction_repair.resting_scale.propose` addresses one monocular ambiguity:
a uniformly scaled object mesh and equally scaled camera-frame translations
have the same pinhole image projection. A declared resting phase and an
independently supported metric floor can constrain this ambiguity.

Supply metric mesh vertices, rigid camera-frame poses, a unit oriented static
floor plane, explicit native frame IDs and a predeclared fit/held-out split.
At least three fit and three held-out observations are required. The proposal
uses the median fit contact scale and separately reports held-out floor error,
fit disagreement and the explicit scale bound. It neither clips a rejected
scale nor mutates its inputs. Rotations stay exact. Returned poses and scaled
vertices are a new paired hypothesis; do not keep old translations unchanged.

Projection preservation is a mathematical property, not new observational
support for metric accuracy. A hypothesis can pass the resting-plane test while
failing foreground-depth or human/object interaction checks. Preserve those
contradictions. Do not infer a resting phase from a depth prediction, spread a
sampled phase to unreviewed frames, overwrite source depth or automatically
approve a complete trajectory. Fresh native image/depth/support/contact checks
and physical evaluation are separate stages. The floor itself may be uncertain.

Useful diagnostic conclusions include: a smoother pose need not be better; an
unchanged image overlay need not imply unchanged physical geometry; and a
successful floor fit does not override a rejected depth-based calibration.
An isolated settling improvement is also separate from a settling pass: report
initial displacement/rotation and late drift independently. A relaxed physical
pose must be projected back into the source images before it is considered as
a target correction. Quiet simulation after settling does not establish that
the changed pose matches the video. Preserve partial physical improvements as
diagnostic evidence when source or initial-support checks still fail.
Eight focused tests cover projection invariance, fit/held-out failures, bounded
scale, invalid evidence and input-preserving float32 handling.
