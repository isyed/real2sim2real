"""Select a temporally coherent path through already admissible pose hypotheses.

This is a declared selection prior, not evidence of hidden motion or an
acceptance override. Native observations, held-out frames and complete joins
still need validation after selection.
"""
import numpy as np


def select(times_s,candidates,*,angular_rate_cost=.01):
    from scipy.spatial.transform import Rotation
    times=np.asarray(times_s,float)
    if (times.ndim!=1 or len(times)<2 or len(candidates)!=len(times)
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or isinstance(angular_rate_cost,bool) or not np.isfinite(angular_rate_cost) or angular_rate_cost<0):
        raise ValueError('Increasing source times and a finite explicit angular-rate cost required')
    levels=[]
    for group in candidates:
        level=[]
        for index,row in enumerate(group):
            if row.get('admissible') is not True:continue
            p=np.asarray(row['pose_camera'],float);score=row['observation_error']
            if (p.shape!=(4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1])
                    or not np.allclose(p[:3,:3]@p[:3,:3].T,np.eye(3),atol=1e-4)
                    or not np.isclose(np.linalg.det(p[:3,:3]),1,atol=1e-4)
                    or isinstance(score,bool) or not np.isfinite(score) or not 0<=score<=1):
                raise ValueError('Admissible rigid poses and observation errors in [0,1] required')
            level.append((index,p,float(score)))
        if not level:raise ValueError('Every anchor requires an independently admissible hypothesis')
        levels.append(level)
    costs=np.array([row[2] for row in levels[0]]);parents=[]
    for i,level in enumerate(levels[1:],1):
        previous=levels[i-1];matrix=np.empty((len(level),len(previous)))
        for j,(_,p,score) in enumerate(level):
            angles=np.array([np.degrees(Rotation.from_matrix(p[:3,:3]@q[:3,:3].T).magnitude()) for _,q,_ in previous])
            matrix[j]=costs+score+angular_rate_cost*angles/(times[i]-times[i-1])
        back=matrix.argmin(1);costs=matrix[np.arange(len(level)),back];parents.append(back)
    chosen=[int(costs.argmin())]
    for back in reversed(parents):chosen.append(int(back[chosen[-1]]))
    chosen.reverse();poses=np.stack([level[i][1] for level,i in zip(levels,chosen)])
    return poses,dict(schema='real2sim2real.anchor-path-selection/1',
        selected_candidate_indices=[int(level[i][0]) for level,i in zip(levels,chosen)],
        angular_rate_cost=float(angular_rate_cost),cost=float(costs.min()),
        selection_only=True,admissibility_overridden=False,hidden_motion_established=False,
        reference_accepted=False,requires_independent_validation=True)
