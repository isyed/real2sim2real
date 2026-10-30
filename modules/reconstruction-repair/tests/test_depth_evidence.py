import numpy as np
import pytest
from real2sim2real.reconstruction_repair.depth_evidence import qualify_depth,DepthEvidenceConfig


def inputs():
    z=np.full((40,40),2.5);obj=np.zeros_like(z,bool);obj[5:35,5:35]=True
    z[10:30,5:35]=5.;occ=np.zeros_like(obj);occ[10:15,5:35]=True
    return z,obj,occ


def run(z,obj,occ,**kw):
    return qualify_depth(z,obj,occ,frame_id=5,anchor_before=(0,2.5),anchor_after=(10,2.5),anchor_review='reviewed RGB/mask/depth',**kw)


def test_outliers_become_missing_without_changing_anchors_or_occluders():
    z,obj,occ=inputs();before=z.copy();out,r=run(z,obj,occ)
    assert r['changed'] and np.array_equal(z,before)
    assert np.all(out[obj&~occ&(z==5)]==0)
    assert np.array_equal(out[~obj|occ],z[~obj|occ])
    assert np.all(out[obj&(z==2.5)]==2.5)
    assert not r['surface_depth_invented'] and not r['approved_for_use']


def test_insufficient_foreground_preserves_every_pixel():
    z,obj,occ=inputs();z[obj]=5.
    out,r=run(z,obj,occ)
    assert not r['changed'] and r['status']=='insufficient_consistent_foreground'
    assert np.array_equal(out,z)


def test_clean_frame_is_preserved():
    z,obj,occ=inputs();z[:]=2.5
    out,r=run(z,obj,occ)
    assert not r['changed'] and np.array_equal(out,z)


@pytest.mark.parametrize('before,after,review', [((0,2.5),(10,5.),'review'),((6,2.5),(10,2.5),'review'),((0,2.5),(10,2.5),'')])
def test_unreviewed_or_inconsistent_temporal_evidence_is_rejected(before,after,review):
    z,obj,occ=inputs()
    with pytest.raises(ValueError):
        qualify_depth(z,obj,occ,frame_id=5,anchor_before=before,anchor_after=after,anchor_review=review)


def test_nonfinite_depth_or_invalid_config_is_rejected():
    z,obj,occ=inputs();z[0,0]=np.nan
    with pytest.raises(ValueError):run(z,obj,occ)
    with pytest.raises(ValueError):DepthEvidenceConfig(min_foreground_fraction=1).validate()
