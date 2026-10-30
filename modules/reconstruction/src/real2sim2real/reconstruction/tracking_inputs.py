"""Rebuild a prediction-only tracking export from a portable reconstruction.

Useful after restoring published artifact members without the original engine's
large intermediate image cache. Camera, masks, depth, mesh and video identities
are explicit. No pose sequence is supplied to registration or tracking.
"""
from pathlib import Path
import io,os,shutil
import numpy as np
from real2sim2real.artifacts import load,role_path,digest,write_json

CAMERA='front_stereo_camera_left'


def link_or_copy(source,target,hardlink):
    target.parent.mkdir(parents=True,exist_ok=True)
    if Path(source).is_symlink():raise ValueError('Tracking inputs cannot depend on symlinks')
    if hardlink:os.link(source,target)
    else:shutil.copy2(source,target)


def create(source,video,output,*,scale=1.,rgb_h5=None,hardlink=False):
    import h5py,trimesh
    from PIL import Image
    source,video,output=map(Path,[source,video,output])
    if output.exists():raise FileExistsError(output)
    artifact=load(source,kinds={'reconstruction'})
    if digest(video)!=artifact['metadata']['source_video_sha256']:raise ValueError('Source RGB identity mismatch')
    with np.load(role_path(source,'geometry'),allow_pickle=False) as z:g=dict(z)
    n=len(g['frame_ids']);K=g['intrinsics'][0]
    if (not np.array_equal(g['frame_ids'],np.arange(n)) or not np.allclose(g['intrinsics'],K)
        or not np.allclose(g['world_to_camera'],np.eye(4))):
        raise ValueError('Tracker requires the complete camera-frame timeline with constant intrinsics')
    if not np.isfinite(scale) or scale<=0:raise ValueError('Finite positive size hypothesis required')
    with h5py.File(role_path(source,'object_masks')) as h:
        shape=h['frames'].shape
        if shape[0]!=n:raise ValueError('Object masks do not cover the source timeline')
    export=output/'export/source';export.mkdir(parents=True)
    for role,rel in [('object_masks',f'object_masks/{CAMERA}.h5'),('human_masks',f'human_masks/{CAMERA}.h5')]:
        path=role_path(source,role)
        with h5py.File(path) as h:
            if h['frames'].shape!=shape or not h.attrs.get('complete'):raise ValueError('Mask coverage incomplete')
        link_or_copy(path,export/rel,hardlink)
    link_or_copy(role_path(source,'depth'),output/'depth.h5',hardlink)
    link_or_copy(video,output/'video.mp4',hardlink)
    target=export/f'images/{CAMERA}.h5';target.parent.mkdir()
    if rgb_h5 is not None:
        with h5py.File(rgb_h5) as h:
            if len(h['frames'])!=n or not h.attrs.get('complete'):raise ValueError('RGB cache coverage incomplete')
        link_or_copy(rgb_h5,target,hardlink)
    else:
        import imageio_ffmpeg
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24',output_params=['-vsync','0']);meta=next(reader)
        width,height=meta['size']
        if (height,width)!=shape[1:]:raise ValueError('RGB and masks have different native dimensions')
        count=0
        with h5py.File(target,'w') as h:
            d=h.create_dataset('frames',shape=(n,),dtype=h5py.vlen_dtype(np.dtype('uint8')))
            d.attrs.update(encoding='jpeg',format='JPEG',colorspace='RGB',height=height,width=width,channels=3,source_dtype='uint8',quality=100,subsampling='4:4:4',optimize=True)
            h.attrs.update(complete=False,sequence='source',camera_id=0)
            for i,frame in enumerate(reader):
                if i>=n:raise ValueError('RGB has more frames than the reconstruction')
                image=Image.frombytes('RGB',(width,height),frame);buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=100,subsampling=0,optimize=True)
                d[i]=np.frombuffer(buffer.getvalue(),dtype=np.uint8);count+=1
            if count!=n:raise ValueError('RGB has fewer frames than the reconstruction')
            h.attrs.modify('complete',True)
    mesh=trimesh.Trimesh(g['vertices']*float(g['object_scale'])*scale,g['faces'],process=False)
    meshpath=export/'object_mesh/output_aligned.glb';meshpath.parent.mkdir();mesh.export(meshpath)
    camera=dict(intrinsics=dict(focal=[float(K[0,0]),float(K[1,1])],principal=[float(K[0,2]),float(K[1,2])]),transform=np.eye(4)[:3].tolist())
    write_json(export/'edex',[dict(cameras=[camera])])
    identity=np.eye(4).tolist()
    write_json(export/'wild_export.json',dict(schema='cari4d.mhr_wild_export.v2',sequence='source',camera_id=0,
        frame_count=n,height=shape[1],width=shape[2],intrinsics=K.tolist(),object_mesh_file=str(meshpath),
        object_pose_frame='centered_axis_aligned',object_pose_frame_revision='cari4d.object_pose_frame.centered_axis_aligned.v1',
        object_pose_storage_frame='output_aligned_mesh_frame',object_mesh_to_training_transform=identity,
        object_pose_storage_to_training_transform=identity,aligned_object_extents_m=mesh.extents.tolist(),
        source_video_sha256=digest(video),parent_artifact_sha256=digest(source/'artifact.json'),resize_hypothesis=scale))
    receipt=dict(frames=n,video_sha256=digest(video),parent_artifact_sha256=digest(source/'artifact.json'),
        scale_hypothesis=scale,old_poses_exported=False,clean_trajectory_used=False,rgb_cache_reused=rgb_h5 is not None,
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    write_json(output/'input_receipt.json',receipt)
    return receipt
