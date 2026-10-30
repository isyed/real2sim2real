# Independent validation of external starting poses

An initial registration can fail even when a nearby image-supported pose exists.
The reconstruction module accepts a bounded set of alternative camera-frame
poses, including hypotheses fitted by `reconstruction-repair.initial_pose_fit`.
This avoids repeating the same model-ranked registration search.

Save one to 32 rigid 4×4 transforms in an NPZ array called `poses_camera`, in
metres, using the exact mesh coordinates of the prediction-only export. Run:

```bash
python -m real2sim2real.reconstruction.initial_registration \
  --export /data/export/source --depth /data/depth.h5 \
  --support /data/support.json --candidate-poses /data/hypotheses.npz \
  --output /data/initial-validation
```

The module measures these poses with the actual GPU silhouette renderer and the
same image, predicted-depth and support gates as ordinary registration. A CPU
optimizer's acceptance flag is never trusted. Invalid/nonrigid arrays fail
before model execution. The proposal file hash is recorded in the report;
`seed.json` is written only if a candidate passes. No later frames are inspected.

This stage validates a starting pose only. It does not certify metric mesh size,
human alignment, a complete trajectory, successful contact or physical stability.
Keep metric calibration and downstream tracking/physics evidence separate.
