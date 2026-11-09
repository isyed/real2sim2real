"""CPU, full-timeline RGB/object/human overlays with a decode receipt."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
import json
import numpy as np
from .artifacts import digest, write_json


def check_display_dependencies(object_display_faces):
    """Fail in the parent before encoding if optional decimation is unavailable."""
    if object_display_faces is None:
        return
    import importlib
    try:
        module=importlib.import_module('fast_simplification')
        if not callable(getattr(module,'simplify',None)):
            raise ImportError('Missing simplify function')
    except ImportError as exc:
        raise RuntimeError('Object display simplification requires fast-simplification in '
            'the rendering environment; install real2sim2real[visualization] or omit '
            'object_display_faces. Numerical source geometry is unchanged.') from exc


def available_pose_ids(value, count):
    """Validate explicitly observed subsets without filling tracking gaps."""
    if value is None:return None
    ids=np.asarray(value)
    if (type(count) is not int or count<1 or ids.ndim!=1
            or len(ids) and (ids.dtype.kind not in 'iu' or np.any(ids<0)
                or np.any(ids>=count) or np.any(np.diff(ids.astype(np.int64))<=0))):
        raise ValueError('Available poses require ordered unique integer IDs inside candidate coverage')
    return ids.astype(np.int64).tolist()


def _initialize(rows, size, fps, title, object_display_faces=None):
    global _STATE
    prepared=[]
    for row in rows:
        with np.load(row['geometry'],allow_pickle=False) as z:g=dict(z)
        if object_display_faces is not None and len(g['faces'])>object_display_faces:
            import trimesh
            display=trimesh.Trimesh(g['vertices'],g['faces'],process=False).simplify_quadric_decimation(face_count=object_display_faces)
            g=dict(g,vertices=np.asarray(display.vertices),faces=np.asarray(display.faces))
        available=row.get('available_frame_ids')
        prepared.append((row['label'],g,np.load(Path(row['surface'])/'vertices.npy',mmap_mode='r'),np.load(Path(row['surface'])/'faces.npy',allow_pickle=False),frozenset(row.get('uncertain_frame_ids',[])),None if available is None else frozenset(available)))
    _STATE=prepared,size,fps,title,object_display_faces


def _render(item):
    from PIL import Image,ImageDraw,ImageFont
    from real2sim2real.reconstruction_repair.geometry import mesh_silhouette
    rows,(width,height),fps,title,object_display_faces=_STATE;i,raw=item
    w=512;h=round(height*w/width);h+=h%2
    rgb=np.asarray(Image.frombytes('RGB',(width,height),raw).resize((w,h)))
    canvas=Image.new('RGB',((len(rows)+1)*w,h+96),'#14202b');canvas.paste(Image.fromarray(rgb),(0,96))
    d=ImageDraw.Draw(canvas);font=ImageFont.load_default()
    d.text((10,12),'Source RGB',fill='white',font=font)
    for col,(label,g,human,faces,uncertain,available) in enumerate(rows,1):
        if i >= len(g['frame_ids']) or available is not None and i not in available:
            canvas.paste(Image.fromarray(rgb),(col*w,96))
            d.text((col*w+10,12),label,fill='white',font=font)
            d.text((col*w+10,29),'NO CANDIDATE POSE — tracking stopped / unavailable',fill='#ff9e8d',font=font)
            continue
        k=g['intrinsics'][i]*np.array([[w/width]*3,[h/height]*3,[1]*3])
        common=dict(intrinsics=k[None],world_to_camera=g['world_to_camera'][i:i+1])
        body=mesh_silhouette(dict(common,vertices=human[i],faces=faces,object_scale=np.array(1.),object_rotation=np.eye(3)[None],object_translation=np.zeros((1,3))),0,h,w)
        obj=mesh_silhouette(dict(common,vertices=g['vertices'],faces=g['faces'],object_scale=g['object_scale'],object_rotation=g['object_rotation'][i:i+1],object_translation=g['object_translation'][i:i+1]),0,h,w)
        panel=rgb.copy();panel[body]=(panel[body]*.65+np.array([40,215,225])*.35).astype(np.uint8)
        panel[obj]=(panel[obj]*.45+np.array([255,140,30])*.55).astype(np.uint8)
        canvas.paste(Image.fromarray(panel),(col*w,96));d.text((col*w+10,12),label,fill='white',font=font)
        if i in uncertain:
            d.text((col*w+10,29),'UNCERTAIN POSE: not observation-validated',fill='#ff9e8d',font=font)
    d.text((10,45),f'{title} | Frame {i} | {i/fps:.2f} s',fill='white',font=font)
    d.text((10,68),'Cyan: full human surface. Orange: object. Transparent overlays; no simulated contact or occlusion.'+
        (' Object display simplified; numerical mesh unchanged.' if object_display_faces is not None else ''),fill='white',font=font)
    return i,np.asarray(canvas)


def render(video, output, candidates, *, title='Reconstruction comparison', workers=4,
           allow_partial=False,object_display_faces=None):
    """Render registered camera/world geometry; caller decodes human surfaces first.

    All candidates must have the complete same zero-based, constant-rate source
    timeline by default. With explicit allow_partial=True, a candidate may cover
    a zero-based prefix; later panels show RGB and an unavailable-pose notice,
    never a carried/interpolated object. Human surfaces retain the full source
    timeline. Input and output hashes are retained. Partial output is not reused.
    ``available_frame_ids`` can explicitly mark disjoint measured intervals on
    that zero-based timeline. Other frames show RGB and an unavailable notice;
    a later reacquisition does not validate or fill the intervening gap.
    """
    import imageio_ffmpeg
    from PIL import Image
    video,output=Path(video),Path(output)
    if type(allow_partial) is not bool:
        raise ValueError('allow_partial must be an explicit boolean')
    if object_display_faces is not None and (type(object_display_faces) is not int or object_display_faces<100):
        raise ValueError('Object display face budget must be None or an integer of at least 100')
    if type(workers) is not int or not 1<=workers<=8 or not 1<=len(candidates)<=3:
        raise ValueError('Use 1–8 workers and 1–3 candidates')
    check_display_dependencies(object_display_faces)
    rows=[]
    for c in candidates:
        uncertain=np.asarray(c.get('uncertain_frame_ids',[]))
        if (uncertain.ndim!=1 or len(uncertain) and (uncertain.dtype.kind not in 'iu'
                or np.any(uncertain<0) or np.any(np.diff(uncertain.astype(np.int64))<=0))):
            raise ValueError('Uncertain poses require ordered unique integer source-frame IDs')
        with np.load(c['geometry'],allow_pickle=False) as g:
            available=available_pose_ids(c.get('available_frame_ids'),len(g['frame_ids']))
            if not allow_partial and available is not None and len(available)!=len(g['frame_ids']):
                raise ValueError('Disjoint availability requires explicit allow_partial=True')
        rows.append(dict(label=str(c['label']),geometry=str(Path(c['geometry']).resolve()),
            surface=str(Path(c['surface']).resolve()),uncertain_frame_ids=uncertain.astype(int).tolist(),available_frame_ids=available))
    identity=dict(video=digest(video),title=title,renderer=digest(__file__),allow_partial=allow_partial,object_display_faces=object_display_faces,candidates=[dict(label=c['label'],geometry=digest(c['geometry']),surface=digest(Path(c['surface'])/'surface.json'),vertices=digest(Path(c['surface'])/'vertices.npy'),faces=digest(Path(c['surface'])/'faces.npy'),uncertain_frame_ids=c['uncertain_frame_ids'],available_frame_ids=c['available_frame_ids']) for c in rows])
    receipt=output/'video_verification.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity']!=identity or saved['sha256']!=digest(output/'comparison.mp4'):
            raise ValueError('Existing video identity differs; preserve it')
        return saved
    if (output/'comparison.partial.mp4').exists():raise ValueError('Preserve partial render before retry')
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);fps=meta['fps'];n=None;coverage=[]
    for c in rows:
        with np.load(c['geometry'],allow_pickle=False) as g:
            count=len(g['frame_ids'])
            if any(frame>=count for frame in c['uncertain_frame_ids']):
                reader.close();raise ValueError('Uncertain pose ID is outside candidate coverage')
            if count<1 or not np.array_equal(g['frame_ids'],np.arange(count)) or not np.allclose(g['timestamps_s'],np.arange(count)/fps,atol=1e-5):
                reader.close();raise ValueError('Candidate/source timeline differs')
        surface=json.loads((Path(c['surface'])/'surface.json').read_text())
        total=surface['frames']
        if ((not allow_partial and total!=count) or count>total
                or (n is not None and n!=total)):
            reader.close();raise ValueError('Surface/candidate timeline differs')
        n=total;coverage.append(count)
    output.mkdir(parents=True,exist_ok=True);w,h=meta['size'];panel_height=round(h*512/w);panel_height+=panel_height%2
    writer=imageio_ffmpeg.write_frames(str(output/'comparison.partial.mp4'),((len(rows)+1)*512,panel_height+96),fps=fps,codec='libx264',quality=7,macro_block_size=1,output_params=['-preset','veryfast','-threads','2','-movflags','+faststart']);writer.send(None)
    completed=0
    def emit(result):
        nonlocal completed
        i,pixels=result
        if i!=completed:raise ValueError('Render reordered source frames')
        writer.send(pixels);completed+=1
        if i in (0,n//2,n-1):Image.fromarray(pixels).save(output/f'frame_{i:06d}.jpg',quality=90)
        if completed%25==0:write_json(output/'render_progress.json',dict(completed=completed,total=n))
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(rows,meta['size'],fps,title,object_display_faces)) as pool:
            pending=deque()
            for i,frame in enumerate(reader):
                if i>=n:raise ValueError('Extra source frames')
                pending.append(pool.submit(_render,(i,frame)))
                if len(pending)>=workers*2:emit(pending.popleft().result())
            while pending:emit(pending.popleft().result())
    finally:reader.close();writer.close()
    if completed!=n:raise ValueError('Incomplete source/video')
    decoder=imageio_ffmpeg.read_frames(str(output/'comparison.partial.mp4'));decoded_meta=next(decoder);decoded=sum(1 for _ in decoder);decoder.close()
    if decoded!=n:raise ValueError('Encoded video does not decode completely')
    (output/'comparison.partial.mp4').replace(output/'comparison.mp4')
    available_counts=[count if c['available_frame_ids'] is None else len(c['available_frame_ids']) for count,c in zip(coverage,rows)]
    result=dict(identity=identity,frames=n,fps=decoded_meta['fps'],sha256=digest(output/'comparison.mp4'),physics=False,
                candidate_coverage_frames=coverage,candidate_available_frames=available_counts,
                complete_candidate_timelines=all(count==n for count in available_counts),
                unavailable_poses_carried_or_interpolated=False,
                candidate_uncertain_frame_ids=[c['uncertain_frame_ids'] for c in rows],
                uncertain_poses_labeled=True)
    result['object_display_faces']=object_display_faces
    result['numerical_source_mesh_modified']=False
    write_json(receipt,result);write_json(output/'render_progress.json',dict(completed=n,total=n,state='complete'))
    return result
