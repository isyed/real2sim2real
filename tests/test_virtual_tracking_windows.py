import shutil
from pathlib import Path
import h5py
import numpy as np
import pytest
from real2sim2real.reconstruction.tracking_inputs import virtual_slice_h5

def source(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(path,'w') as f:
        f.attrs['complete']=True
        f.create_dataset('frames',data=np.arange(24,dtype=np.uint8).reshape(4,3,2)).attrs['kind']='mask'
        d=f.create_dataset('raw/camera',(4,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i in range(4):d[i]=np.arange(i+2,dtype=np.uint8)
        f.create_dataset('alignment/scale',data=np.arange(4,dtype=np.float32))
        f.create_dataset('frame_names/camera',data=[b'a',b'b',b'c',b'd'])

def test_backward_masks_payloads_and_names_are_exact(tmp_path):
    p=tmp_path/'base.h5';q=tmp_path/'window/result.h5';source(p)
    r=virtual_slice_h5(p,q,[3,2,1],4)
    with h5py.File(p) as a,h5py.File(q) as b:
        assert b['frames'].is_virtual and b.attrs['complete']
        assert b['frames'].attrs['kind']=='mask'
        for j,i in enumerate([3,2,1]):
            np.testing.assert_array_equal(a['frames'][i],b['frames'][j])
            np.testing.assert_array_equal(a['raw/camera'][i],b['raw/camera'][j])
        assert list(b['frame_names/camera'][:])==[b'000000',b'000001',b'000002']
        np.testing.assert_array_equal(b['alignment/scale'][:],[3,2,1])
    assert r['source_retention_required'] and not r['payload_copied']
    assert not Path(r['source_relative']).is_absolute()

def test_bundle_can_move_without_original_absolute_path(tmp_path):
    root=tmp_path/'first';p=root/'shared/base.h5';q=root/'windows/a.h5';source(p)
    virtual_slice_h5(p,q,[1,2],4)
    moved=tmp_path/'moved';shutil.copytree(root,moved);shutil.rmtree(root)
    with h5py.File(moved/'windows/a.h5') as f:np.testing.assert_array_equal(f['frames'][0],np.arange(6,12,dtype=np.uint8).reshape(3,2))

def test_scattered_indices_are_not_silently_retimed(tmp_path):
    p=tmp_path/'a.h5';source(p)
    with pytest.raises(ValueError,match='contiguous'):virtual_slice_h5(p,tmp_path/'b.h5',[0,2],4)

def test_incomplete_source_rejected(tmp_path):
    p=tmp_path/'a.h5';source(p)
    with h5py.File(p,'r+') as f:f.attrs['complete']=False
    with pytest.raises(ValueError,match='complete'):virtual_slice_h5(p,tmp_path/'b.h5',[0,1],4)
