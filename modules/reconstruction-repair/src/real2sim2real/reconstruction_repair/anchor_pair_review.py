"""Rank already-admissible registration pairs using observed intermediate RGB.

Depth is intentionally not used to rank when independently diagnosed as corrupt.
This does not waive later depth/metric gates. Held-out frames cannot choose pairs.
"""
import time
import numpy as np
from .anchor_hypothesis import initialize
from .geometry import mesh_silhouette


def rank(data, first_candidates, last_candidates, *, fps, fit_ids, heldout_ids,
         max_seconds=120., max_pairs=256, finalists=3):
    ids=np.asarray(data['frame_ids']);fit=np.asarray(fit_ids);held=np.asarray(heldout_ids)
    if (len(fit)<3 or len(held)<2 or np.intersect1d(fit,held).size
            or not np.isin(np.r_[fit,held],ids[1:-1]).all()
            or max_seconds<=0 or max_pairs<1 or finalists<1):
        raise ValueError('Disjoint observed intermediate fit/held-out IDs and finite budget required')
    if len(first_candidates)*len(last_candidates)>max_pairs:
        raise ValueError('Candidate pair count exceeds declared budget')
    started=time.monotonic();results=[]
    def assess(poses,subset):
        g=dict(vertices=data['vertices'],faces=data['faces'],object_scale=np.array(1.),
            object_rotation=poses[:,:3,:3],object_translation=poses[:,:3,3],
            intrinsics=data['intrinsics'],world_to_camera=np.tile(np.eye(4),(len(ids),1,1)))
        rows=[]
        for frame in subset:
            i=int(np.flatnonzero(ids==frame)[0]);obj=data['object_mask'][i]&~data['occluder_mask'][i]
            if not obj.any():raise ValueError('Unobserved frame cannot rank or validate a pair')
            silhouette=mesh_silhouette(g,i,*obj.shape)&~data['occluder_mask'][i]
            rows.append(dict(source_frame=int(frame),iou=float((silhouette&obj).sum()/max(1,(silhouette|obj).sum()))))
        return rows
    for i,a in enumerate(first_candidates):
        for j,b in enumerate(last_candidates):
            if time.monotonic()-started>max_seconds:break
            poses,_=initialize(ids,ids[[0,-1]],np.asarray([a,b]),fps=fps)
            rows=assess(poses,fit)
            results.append(dict(first_index=i,last_index=j,fit=rows,fit_mean_iou=float(np.mean([r['iou'] for r in rows]))))
        if time.monotonic()-started>max_seconds:break
    ordered=sorted(results,key=lambda r:r['fit_mean_iou'],reverse=True)
    for row in ordered[:finalists]:
        poses,_=initialize(ids,ids[[0,-1]],np.asarray([first_candidates[row['first_index']],last_candidates[row['last_index']]]),fps=fps)
        row['heldout']=assess(poses,held);row['heldout_mean_iou']=float(np.mean([r['iou'] for r in row['heldout']]))
    return dict(schema='real2sim2real.anchor-pair-review/1',ranked=ordered,
        planned_pairs=len(first_candidates)*len(last_candidates),tested_pairs=len(results),
        complete=len(results)==len(first_candidates)*len(last_candidates),seconds=time.monotonic()-started,
        selection_uses='fit_RGB_silhouettes_only',hypothesis_only=True,depth_validated=False,
        full_reference_accepted=False,metric_truth_known=False)
