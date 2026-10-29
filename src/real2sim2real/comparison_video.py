"""CPU, full-timeline RGB/object/human overlays with a decode receipt."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import multiprocessing
import json
import numpy as np
from .artifacts import digest, write_json


def _initialize(rows, size, fps, title):
    global _STATE
    prepared=[]
    for row in rows:
        with np.load(row['geometry'],allow_pickle=False) as z:g=dict(z)
        prepared.append((row['label'],g,np.load(Path(row['surface'])/'vertices.npy',mmap_mode='r'),np.load(Path(row['surface'])/'faces.npy',allow_pickle=False)))
    _STATE=prepared,size,fps,title


def _render(item):
    from PIL import Image,ImageDraw,ImageFont
    from real2sim2real.reconstruction_repair.geometry import mesh_silhouette
    rows,(width,height),fps,title=_STATE;i,raw=item
    w=512;h=round(height*w/width);h+=h%2
    rgb=np.asarray(Image.frombytes('RGB',(width,height),raw).resize((w,h)))
    canvas=Image.new('RGB',((len(rows)+1)*w,h+96),'#14202b');canvas.paste(Image.fromarray(rgb),(0,96))
    d=ImageDraw.Draw(canvas);font=ImageFont.load_default()
    d.text((10,12),'Source RGB',fill='white',font=font)
    for col,(label,g,human,faces) in enumerate(rows,1):
        k=g['intrinsics'][i]*np.array([[w/width]*3,[h/height]*3,[1]*3])
        common=dict(intrinsics=k[None],world_to_camera=g['world_to_camera'][i:i+1])
        body=mesh_silhouette(dict(common,vertices=human[i],faces=faces,object_scale=np.array(1.),object_rotation=np.eye(3)[None],object_translation=np.zeros((1,3))),0,h,w)
        obj=mesh_silhouette(dict(common,vertices=g['vertices'],faces=g['faces'],object_scale=g['object_scale'],object_rotation=g['object_rotation'][i:i+1],object_translation=g['object_translation'][i:i+1]),0,h,w)
        panel=rgb.copy();panel[body]=(panel[body]*.65+np.array([40,215,225])*.35).astype(np.uint8)
        panel[obj]=(panel[obj]*.45+np.array([255,140,30])*.55).astype(np.uint8)
        canvas.paste(Image.fromarray(panel),(col*w,96));d.text((col*w+10,12),label,fill='white',font=font)
    d.text((10,45),f'{title} | Frame {i} | {i/fps:.2f} s',fill='white',font=font)
    d.text((10,68),'Cyan: full human surface. Orange: object. Transparent overlays; no simulated contact or occlusion.',fill='white',font=font)
    return i,np.asarray(canvas)


def render(video, output, candidates, *, title='Reconstruction comparison', workers=4):
    """Render registered camera/world geometry; caller decodes human surfaces first.

    All candidates must have the complete same zero-based, constant-rate source
    timeline. Input and output hashes are retained. Partial output is not reused.
    """
    import imageio_ffmpeg
    from PIL import Image
    video,output=Path(video),Path(output)
    if type(workers) is not int or not 1<=workers<=8 or not 1<=len(candidates)<=3:
        raise ValueError('Use 1–8 workers and 1–3 candidates')
    rows=[dict(label=str(c['label']),geometry=str(Path(c['geometry']).resolve()),surface=str(Path(c['surface']).resolve())) for c in candidates]
    identity=dict(video=digest(video),title=title,renderer=digest(__file__),candidates=[dict(label=c['label'],geometry=digest(c['geometry']),surface=digest(Path(c['surface'])/'surface.json'),vertices=digest(Path(c['surface'])/'vertices.npy'),faces=digest(Path(c['surface'])/'faces.npy')) for c in rows])
    receipt=output/'video_verification.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity']!=identity or saved['sha256']!=digest(output/'comparison.mp4'):
            raise ValueError('Existing video identity differs; preserve it')
        return saved
    if (output/'comparison.partial.mp4').exists():raise ValueError('Preserve partial render before retry')
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);fps=meta['fps'];n=None
    for c in rows:
        with np.load(c['geometry'],allow_pickle=False) as g:
            count=len(g['frame_ids'])
            if not np.array_equal(g['frame_ids'],np.arange(count)) or not np.allclose(g['timestamps_s'],np.arange(count)/fps,atol=1e-5):
                reader.close();raise ValueError('Candidate/source timeline differs')
        surface=json.loads((Path(c['surface'])/'surface.json').read_text())
        if surface['frames']!=count or (n is not None and n!=count):reader.close();raise ValueError('Surface/candidate timeline differs')
        n=count
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
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(rows,meta['size'],fps,title)) as pool:
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
    result=dict(identity=identity,frames=n,fps=decoded_meta['fps'],sha256=digest(output/'comparison.mp4'),physics=False)
    write_json(receipt,result);write_json(output/'render_progress.json',dict(completed=n,total=n,state='complete'))
    return result
