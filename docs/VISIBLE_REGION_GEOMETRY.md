# Visible geometry versus a settled collision pose

Physical settling can create a stable collision configuration that disagrees
with the observed object. `egocentric.visible_region_geometry` compares fixed
original and settled-offset pose hypotheses against image-defined surface
regions, preserving both inputs. It never fits a correction or infers hidden
contact normals from visible top surfaces.

```bash
python -m real2sim2real.egocentric.visible_region_geometry \
  --config /data/visible-region-audit.json
```

Supply an accepted metric `reconstruction`, matching upright `support`, original
`capture`, accepted `calibration`, accepted `settling` receipt, hash-pinned
`sources`, separate `fit_frames` and `validation_frames`, fresh `output` and a
finite `deadline_s`. These are reconstruction frame indices; the output retains
capture frame IDs and timestamps. There is no optimization even on the fit split.

The audit uses the reconstruction's object mask and two hand-occluder masks.
`regions` declares `lower_fraction`, erosion `border` and background `context`
in pixels. It separates the lower object region, upper body, enclosed hole and
adjacent background. These are image proxies, not verified semantic parts or
contact patches. Holes, background and object regions remain disjoint; dilated
hands are excluded. The geometric split is useful only when the desired part
occupies the declared image region; inspect exported overlays before interpreting
its metrics.

`maximum_region_samples` selects deterministic evenly spaced raster positions
across each valid-depth region. Ray casting finds the first visible mesh surface
at the original pose and at a fixed settled-start offset applied to each source
pose. Report both coverage and depth error: common-ray errors compare identical
pixels, while individual ray coverage reveals predictions that miss the object.
Missing rays never count as zero error. The fixed offset is a diagnostic
hypothesis for comparable pre-grasp views, not re-simulated settling at every frame.

The `matching` configuration supplies the existing reciprocal rectified stereo
matcher settings; `maximum_corners` bounds feature trials. Source depth proposes
a region of interest, then textured image corners are matched without mesh-normal
or mesh-depth agreement gates. Features are triangulated with accepted calibration
and assigned to the image regions. Accepted patches can still be ambiguous or
straddle a boundary. A match through an apparent opening at object height should
remain visible as a possible mismatch or occlusion, not be silently discarded.
These checks share stereo inputs and calibration with the reconstruction; they
are consistency evidence, not independent ground truth.

Upright heights of the object, hole and adjacent background help distinguish a
local surface discrepancy from a proposed global table shift. Depth bleed,
reflection and poor texture can contaminate every region. Agreement on visible
upper geometry cannot establish underside thickness, support feet, force-bearing
patches or insertion clearance. Preserve uncertainty until further views or
measurements support a correction, then validate on separate frames before
fitting both hands. No robot fitting, physics rollout or training is performed.

Outputs are `audit.json`, source/implementation hashes, image overlays and
`review.html` with height, paired error and coverage tables. Dependencies are
CPU NumPy/SciPy, trimesh with a ray backend, Pillow, OpenCV and capture decoding;
OpenCV loads only inside the image audit. See [strategy audit](GUIDED_MOTION.md)
and [stereo contact evidence](STEREO_CONTACT_EVIDENCE.md).
