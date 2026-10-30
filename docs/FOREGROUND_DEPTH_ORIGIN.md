# Diagnose foreground depth before changing tracking thresholds

A small within-frame depth spread is not enough to certify a foreground patch.
Its predicted distance can jump between adjacent frames while the RGB object
remains visible. Inspect raw prediction and subsequent human-depth alignment
separately before treating a tracking rejection as a controller failure.

Run the optional input-only audit:

```sh
python -m real2sim2real.reconstruction_repair.foreground_visibility \
  --input /data/reconstruction --output /data/depth-audit --include-raw-depth
```

The artifact must retain native raw and aligned predicted depth, both masks and
the source timeline. The report includes both foreground audits, warnings that
already occur in raw depth, warnings appearing only after alignment, the stored
alignment scales when available, and scene-wide raw depth medians. Raw/aligned
warnings use the same declared temporal bounds. Measurements remain source-frame
indexed, and artifact hashes pin their inputs. Without the option, the original
aligned-depth audit is unchanged.

A local raw-depth jump with stable scene-wide statistics and alignment scale
suggests a foreground prediction/correspondence problem rather than a change
introduced solely by the alignment multiplier. Inspect actual RGB and masks.
Neither diagnosis proves true depth, repairs missing geometry nor authorizes
hidden-pose interpolation, threshold relaxation, physics or actor training.
The audit changes no depth, pose, mesh or selected reference.
