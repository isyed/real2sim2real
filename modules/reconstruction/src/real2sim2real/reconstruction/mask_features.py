"""Explicit image-region coverage checks for thin, task-relevant object parts.

Caller-supplied regions come from RGB inspection. Coverage is a diagnostic,
never automatic semantic approval or evidence of metric/physical accuracy.
"""
from pathlib import Path
import numpy as np
from real2sim2real.artifacts import digest, write_json


def audit(mask, features):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not features:
        raise ValueError('Require a nonempty 2D mask and explicit feature regions')
    mask = mask != 0
    rows=[]
    names=set()
    for feature in features:
        name=feature.get('name');box=feature.get('xyxy');minimum=feature.get('min_fraction',0.)
        maximum=feature.get('max_fraction',1.)
        if (not isinstance(name,str) or not name.strip() or name in names
                or not isinstance(box,list) or len(box)!=4
                or any(type(x) is not int for x in box)
                or not 0<=box[0]<box[2]<=mask.shape[1]
                or not 0<=box[1]<box[3]<=mask.shape[0]
                or not np.isfinite([minimum,maximum]).all() or not 0<=minimum<=maximum<=1):
            raise ValueError('Require unique names, bounded native-pixel regions and fractions')
        names.add(name)
        x0,y0,x1,y1=box;fraction=float(mask[y0:y1,x0:x1].mean())
        rows.append(dict(name=name,xyxy=box,foreground_fraction=fraction,
            min_fraction=float(minimum),max_fraction=float(maximum),
            passed=bool(minimum<=fraction<=maximum)))
    return dict(passed=all(r['passed'] for r in rows),features=rows,
        semantics_verified=False,geometry_validated=False)


def check(output, checks, *, source_video_sha256):
    from PIL import Image
    output=Path(output);rows=[]
    for item in checks:
        frame=item.get('frame_index')
        if type(frame) is not int or frame<0:raise ValueError('Expected native source-frame index')
        path=output/'masks/1'/f'{frame:06d}.png'
        with Image.open(path) as image:
            mask=np.asarray(image.convert('L'))
        result=audit(mask,item['features'])
        rows.append(dict(frame_index=frame,mask_sha256=digest(path),native_size=[mask.shape[1],mask.shape[0]],**result))
    result=dict(schema='real2sim2real.mask-feature-coverage/1',
        source_video_sha256=source_video_sha256,checks=rows,passed=bool(rows) and all(r['passed'] for r in rows),
        semantics_verified=False,geometry_validated=False)
    write_json(output/'mask_review/feature_coverage.json',result)
    return result


def enforce(request):
    checks=request['options'].get('mask_feature_checks')
    if checks is None:return None
    from .mask_acceptance import MaskReviewRequired
    result=check(request['output'],checks,source_video_sha256=request['input_sha256'])
    if not result['passed']:
        raise MaskReviewRequired('Explicit object-feature mask coverage failed; inspect mask_review/feature_coverage.json before mesh inference')
    return result
