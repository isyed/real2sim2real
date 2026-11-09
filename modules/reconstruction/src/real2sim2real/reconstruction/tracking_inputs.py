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


def tracking_mesh(geometry, *, source_mesh=None, scale=1., require_appearance=False):
    """Preserve verified source appearance when rebuilding a tracker mesh.

    RGB registration cannot recover a painted feature if an adapter silently
    discards its vertex colours/UVs. Appearance is reusable only after exact
    topology and metric vertex-frame correspondence have been established.
    """
    import trimesh
    vertices=np.asarray(geometry['vertices'])*float(geometry['object_scale'])
    faces=np.asarray(geometry['faces'])
    if type(require_appearance) is not bool or not np.isfinite(scale) or scale<=0:
        raise ValueError('Explicit appearance requirement and positive scale required')
    status='no_source_appearance'
    mesh=None
    if source_mesh is not None:
        candidate=trimesh.load(source_mesh,force='mesh',process=False)
        matches=(candidate.vertices.shape==vertices.shape and np.array_equal(candidate.faces,faces)
                 and np.allclose(candidate.vertices,vertices,rtol=0,atol=1e-7))
        if matches and candidate.visual.kind in ('vertex','texture','face'):
            mesh=candidate.copy()
            status='verified_source_'+candidate.visual.kind
        elif not matches:
            status='source_mesh_frame_or_topology_mismatch'
    if mesh is None:
        if require_appearance:
            raise ValueError('RGB cue requires verified source mesh appearance: '+status)
        mesh=trimesh.Trimesh(vertices,faces,process=False)
    mesh.vertices=np.asarray(mesh.vertices)*scale
    return mesh,dict(status=status,appearance_preserved=mesh.visual.kind is not None and status.startswith('verified_source_'),
                     source_mesh_sha256=None if source_mesh is None else digest(source_mesh),
                     appearance_required=require_appearance,scale=float(scale))


def link_or_copy(source,target,hardlink):
    target.parent.mkdir(parents=True,exist_ok=True)
    if Path(source).is_symlink():raise ValueError('Tracking inputs cannot depend on symlinks')
    if hardlink:os.link(source,target)
    else:shutil.copy2(source,target)


def window_indices(frame_indices, source_frames):
    """Map local tracker indices to an explicit contiguous source interval."""
    ids = np.asarray(frame_indices)
    if (ids.ndim != 1 or len(ids) < 1 or ids.dtype.kind not in 'iu'
            or np.any(ids < 0) or np.any(ids >= source_frames)
            or (len(ids) > 1 and not (np.all(np.diff(ids.astype(np.int64)) == 1)
                                     or np.all(np.diff(ids.astype(np.int64)) == -1)))):
        raise ValueError('Window must be contiguous source frame indices, forward or backward')
    return ids.astype(np.int64)


def slice_h5(source, target, indices, source_frames):
    """Slice every timeline together, preserving codecs and resetting local names."""
    import h5py
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source) as src, h5py.File(target, 'x') as dst:
        if not src.attrs.get('complete'):
            raise ValueError('Window requires a complete source cache')
        dst.attrs.update(dict(src.attrs)); dst.attrs['complete'] = False
        def visit(name, value):
            if isinstance(value, h5py.Group):
                dst.require_group(name).attrs.update(dict(value.attrs)); return
            if not value.shape or value.shape[0] != source_frames:
                raise ValueError('Unmapped HDF5 dataset: '+name)
            shape = (len(indices), *value.shape[1:])
            options = dict(compression='gzip', compression_opts=1, chunks=(1, *shape[1:])) if len(shape) > 1 else {}
            out = dst.create_dataset(name, shape=shape, dtype=value.dtype, **options)
            out.attrs.update(dict(value.attrs))
            for local, frame in enumerate(indices):
                out[local] = f'{local:06d}' if name.startswith('frame_names/') else value[int(frame)]
        src.visititems(visit)
        dst.attrs['complete'] = True


def virtual_slice_h5(source, target, indices, source_frames):
    """Create a portable zero-copy window over an explicitly retained cache.

    The source must remain available and read-only. The receipt names it using
    a path relative to the window, so a self-contained input folder can move
    without retaining absolute workstation paths. Mount/archive both files.
    Local frame names are materialized; pixel/depth payloads are not copied.
    """
    import h5py
    source,target=Path(source),Path(target)
    if source.is_symlink() or target.exists():raise ValueError('Regular retained source and fresh virtual target required')
    ids=window_indices(indices,source_frames)
    relative=os.path.relpath(source.resolve(),target.parent.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(source) as src:
        if not src.attrs.get('complete'):raise ValueError('Virtual window requires a complete source cache')
        with h5py.File(target,'x',libver='latest') as dst:
            dst.attrs.update(dict(src.attrs));dst.attrs['complete']=False
            def visit(name,value):
                if isinstance(value,h5py.Group):
                    dst.require_group(name).attrs.update(dict(value.attrs));return
                if not value.shape or value.shape[0]!=source_frames:
                    raise ValueError('Unmapped virtual HDF5 dataset: '+name)
                shape=(len(ids),*value.shape[1:])
                if name.startswith('frame_names/'):
                    out=dst.create_dataset(name,shape=shape,dtype=value.dtype)
                    for local in range(len(ids)):out[local]=f'{local:06d}'
                else:
                    layout=h5py.VirtualLayout(shape=shape,dtype=value.dtype)
                    original=h5py.VirtualSource(relative,name,shape=value.shape)
                    for local,index in enumerate(ids):layout[local]=original[int(index)]
                    out=dst.create_virtual_dataset(name,layout)
                out.attrs.update(dict(value.attrs))
            src.visititems(visit);dst.attrs['complete']=True
    receipt=dict(schema='real2sim2real.virtual-tracking-window/1',
        source_relative=relative,source_sha256=digest(source),source_frames=source_frames,
        source_frame_indices=ids.tolist(),local_frame_indices=list(range(len(ids))),
        payload_copied=False,local_frame_names_reset=True,source_retention_required=True,
        source_must_be_mounted=True,window_sha256=digest(target))
    write_json(target.with_suffix(target.suffix+'.sources.json'),receipt)
    return receipt


def create(source,video,output,*,scale=1.,rgb_h5=None,hardlink=False,frame_indices=None,
           require_appearance=False):
    import h5py
    from PIL import Image
    source,video,output=map(Path,[source,video,output])
    if output.exists():raise FileExistsError(output)
    artifact=load(source,kinds={'reconstruction'})
    if digest(video)!=artifact['metadata']['source_video_sha256']:raise ValueError('Source RGB identity mismatch')
    with np.load(role_path(source,'geometry'),allow_pickle=False) as z:g=dict(z)
    source_frames=len(g['frame_ids']);K=g['intrinsics'][0]
    n=source_frames
    if (not np.array_equal(g['frame_ids'],np.arange(n)) or not np.allclose(g['intrinsics'],K)
        or not np.allclose(g['world_to_camera'],np.eye(4))):
        raise ValueError('Tracker requires the complete camera-frame timeline with constant intrinsics')
    if not np.isfinite(scale) or scale<=0:raise ValueError('Finite positive size hypothesis required')
    indices = None if frame_indices is None else window_indices(frame_indices, source_frames)
    if indices is not None:
        intervals=np.diff(g['timestamps_s'])
        if (not len(intervals) or not np.isfinite(intervals).all() or intervals[0]<=0
                or not np.allclose(intervals,intervals[0],atol=1e-7,rtol=1e-5)):
            raise ValueError('Tracking windows require an explicit constant source frame interval')
        source_fps=float(1./intervals[0])
    with h5py.File(role_path(source,'object_masks')) as h:
        shape=h['frames'].shape
        if shape[0]!=n:raise ValueError('Object masks do not cover the source timeline')
    export=output/'export/source';export.mkdir(parents=True)
    for role,rel in [('object_masks',f'object_masks/{CAMERA}.h5'),('human_masks',f'human_masks/{CAMERA}.h5')]:
        path=role_path(source,role)
        with h5py.File(path) as h:
            if h['frames'].shape!=shape or not h.attrs.get('complete'):raise ValueError('Mask coverage incomplete')
        if indices is None:link_or_copy(path,export/rel,hardlink)
        else:slice_h5(path,export/rel,indices,source_frames)
    if indices is None:link_or_copy(role_path(source,'depth'),output/'depth.h5',hardlink)
    else:slice_h5(role_path(source,'depth'),output/'depth.h5',indices,source_frames)
    link_or_copy(video,output/'video.mp4',hardlink)
    target=export/f'images/{CAMERA}.h5';target.parent.mkdir()
    if rgb_h5 is not None:
        with h5py.File(rgb_h5) as h:
            if len(h['frames'])!=n or not h.attrs.get('complete'):raise ValueError('RGB cache coverage incomplete')
        if indices is None:link_or_copy(rgb_h5,target,hardlink)
        else:slice_h5(rgb_h5,target,indices,source_frames)
    else:
        import imageio_ffmpeg
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24',output_params=['-vsync','0']);meta=next(reader)
        width,height=meta['size']
        if (height,width)!=shape[1:]:raise ValueError('RGB and masks have different native dimensions')
        count=0
        with h5py.File(target,'w') as h:
            d=h.create_dataset('frames',shape=(n if indices is None else len(indices),),dtype=h5py.vlen_dtype(np.dtype('uint8')))
            d.attrs.update(encoding='jpeg',format='JPEG',colorspace='RGB',height=height,width=width,channels=3,source_dtype='uint8',quality=100,subsampling='4:4:4',optimize=True)
            h.attrs.update(complete=False,sequence='source',camera_id=0)
            wanted = None if indices is None else {int(v):i for i,v in enumerate(indices)}
            for i,frame in enumerate(reader):
                if i>=n:raise ValueError('RGB has more frames than the reconstruction')
                count+=1
                if wanted is not None and i not in wanted:continue
                image=Image.frombytes('RGB',(width,height),frame);buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=100,subsampling=0,optimize=True)
                d[i if wanted is None else wanted[i]]=np.frombuffer(buffer.getvalue(),dtype=np.uint8)
            if count!=n:raise ValueError('RGB has fewer frames than the reconstruction')
            h.attrs.modify('complete',True)
    if indices is not None:
        n = len(indices)
        write_json(output/'source_frame_mapping.json',dict(schema='real2sim2real.tracking-window/1',
            source_frame_indices=indices.tolist(), local_frame_indices=list(range(n)),
            source_frames=source_frames, tracker_update_count=max(0,n-1), source_video_sha256=digest(video),
            source_fps=source_fps,
            parent_artifact_sha256=digest(source/'artifact.json'),
            full_timeline=False, direction='forward' if n == 1 or indices[1] > indices[0] else 'backward'))
    appearance_source=source/artifact['files']['mesh']['path'] if 'mesh' in artifact['files'] else None
    mesh,appearance=tracking_mesh(g,source_mesh=appearance_source,scale=scale,
                                  require_appearance=require_appearance)
    meshpath=export/'object_mesh/output_aligned.glb';meshpath.parent.mkdir();mesh.export(meshpath)
    camera=dict(intrinsics=dict(focal=[float(K[0,0]),float(K[1,1])],principal=[float(K[0,2]),float(K[1,2])]),transform=np.eye(4)[:3].tolist())
    write_json(export/'edex',[dict(cameras=[camera])])
    identity=np.eye(4).tolist()
    write_json(export/'wild_export.json',dict(schema='cari4d.mhr_wild_export.v2',sequence='source',camera_id=0,
        frame_count=n,height=shape[1],width=shape[2],intrinsics=K.tolist(),object_mesh_file=str(meshpath),
        object_pose_frame='centered_axis_aligned',object_pose_frame_revision='cari4d.object_pose_frame.centered_axis_aligned.v1',
        object_pose_storage_frame='output_aligned_mesh_frame',object_mesh_to_training_transform=identity,
        object_pose_storage_to_training_transform=identity,aligned_object_extents_m=mesh.extents.tolist(),
        source_video_sha256=digest(video),parent_artifact_sha256=digest(source/'artifact.json'),resize_hypothesis=scale,
        mesh_appearance=appearance))
    receipt=dict(frames=n,tracker_update_count=max(0,n-1),video_sha256=digest(video),parent_artifact_sha256=digest(source/'artifact.json'),
        scale_hypothesis=scale,mesh_appearance=appearance,
        old_poses_exported=False,clean_trajectory_used=False,rgb_cache_reused=rgb_h5 is not None,
        full_timeline=indices is None, source_frames=source_frames,
        source_frame_indices=None if indices is None else indices.tolist(),
        files={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()})
    write_json(output/'input_receipt.json',receipt)
    return receipt
