import numpy as np
import pytest
from real2sim2real.human_surface import native_to_world


def test_native_surface_units_axes_and_world_transform():
    native=np.array([[[100.,-200.,-300.],[0.,0.,-100.]]])
    transform=np.eye(4)
    transform[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    transform[:3,3]=[2,3,4]
    result=native_to_world(native,transform)
    np.testing.assert_allclose(result,[[[0,4,7],[2,3,5]]])
    np.testing.assert_array_equal(native,[[[100,-200,-300],[0,0,-100]]])


def test_reject_invalid_surface_or_nonrigid_transform():
    with pytest.raises(ValueError):native_to_world(np.full((1,3,3),np.nan),np.eye(4))
    with pytest.raises(ValueError):native_to_world(np.zeros((3,3)),np.eye(4))
    transform=np.eye(4);transform[0,0]=2
    with pytest.raises(ValueError):native_to_world(np.zeros((1,3,3)),transform)


def test_completed_cache_reuse_without_torch_and_tamper_rejection(tmp_path,monkeypatch):
    import builtins
    import json
    from real2sim2real.human import file_sha256
    from real2sim2real.human_surface import decode_surface
    human=tmp_path/'human.npz';model=tmp_path/'model.pt';out=tmp_path/'surface'
    human.write_bytes(b'producer-owned-input');model.write_bytes(b'producer-owned-model');out.mkdir()
    np.save(out/'vertices.npy',np.zeros((1,3,3),np.float32))
    np.save(out/'faces.npy',np.array([[0,1,2]]))
    receipt=dict(human_sha256=file_sha256(human),model_sha256=file_sha256(model),world_transform=np.eye(4).tolist(),
        vertices_sha256=file_sha256(out/'vertices.npy'),faces_sha256=file_sha256(out/'faces.npy'))
    (out/'surface.json').write_text(json.dumps(receipt))
    original_import=builtins.__import__
    def forbid_torch(name,*args,**kwargs):
        if name=='torch' or name.startswith('torch.'):raise AssertionError('Cache reuse imported Torch')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',forbid_torch)
    assert decode_surface(human,model,out)==receipt
    (out/'vertices.npy').write_bytes(b'changed surface')
    with pytest.raises(ValueError,match='surface changed'):decode_surface(human,model,out)
