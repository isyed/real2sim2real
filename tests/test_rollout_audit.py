import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from real2sim2real.controller_actor_training.rollout_audit import pose_errors,add_summary,motion_screen,phase_labels,phase_summary,intervals


def identity(n=10):
    x=np.zeros((n,7));x[:,6]=1;return x


def test_translation_rotation_and_nonmutation():
    b=identity();a=b.copy();a[:,0]=.05;original=a.copy()
    v=np.array([[1.,0,0],[-1.,0,0]])
    e=pose_errors(a,b,v,chunk_size=3)
    np.testing.assert_allclose(e['add_m'],.05)
    np.testing.assert_allclose(e['rotation_only_add_m'],0)
    np.testing.assert_array_equal(a,original)
    a=identity();a[:,3:]=Rotation.from_euler('z',180,degrees=True).as_quat()
    e=pose_errors(a,b,v)
    np.testing.assert_allclose(e['position_m'],0)
    np.testing.assert_allclose(e['rotation_only_add_m'],2)
    np.testing.assert_allclose(e['orientation_deg'],180)


def test_quaternion_sign_does_not_create_motion_or_error():
    a=identity();b=a.copy();a[::2,3:]*=-1
    np.testing.assert_allclose(pose_errors(a,b,[[1,1,1]])['add_m'],0)
    assert motion_screen(a,fps=30)['rotation_warning_frames']==0


def test_accuracy_strict_threshold_and_normalized_auc():
    x=np.array([0,.05,.1]);t=np.arange(.01,.1,.01)
    r=add_summary(x,t)
    expected=[np.mean(x<th) for th in t]
    assert r['accuracy']==expected
    assert r['auc']==pytest.approx(np.trapezoid(expected,x=np.linspace(0,1,9)))


@pytest.mark.parametrize('bad',[np.full((2,7),np.nan),np.zeros((2,7)),np.ones((7,))])
def test_invalid_pose_rejected(bad):
    with pytest.raises(ValueError):pose_errors(bad,identity(2),[[0,0,0]])


def test_phases_partition_and_weighted_auc_reconstructs_whole_clip():
    r=identity(100);r[30:70,0]=np.arange(40)*.002;r[70:,0]=r[69,0]
    contact=np.zeros(100,bool);contact[20:80]=True
    labels=phase_labels(r,contact,fps=30)
    e=pose_errors(identity(100),r,[[0,0,0]])
    rows=phase_summary(e,labels,np.arange(.01,.1,.01))
    assert sum(x['frames'] for x in rows)==100
    assert sum(x['fraction']*x['auc'] for x in rows)==pytest.approx(add_summary(e['add_m'],np.arange(.01,.1,.01))['auc'])
    assert labels[0]=='approach' and labels[25]=='labelled_contact' and labels[-1]=='late_rest'


def test_native_jump_screen_and_longest_intervals():
    x=identity(10);x[5:,0]=.2
    assert motion_screen(x,fps=30)['warning_frames']==[5]
    r=intervals([0,1,1,0,1,1,1,0],fps=2)
    assert r[0]['start_frame']==4 and r[0]['duration_s']==1.5
