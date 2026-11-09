"""Combine verified saved-physics panels from separately framed policy reviews.

This renders no geometry and runs no simulator. Each input must be a completed
1280x640 reference-versus-physics video produced by ``render_frozen``. Right-hand
physical panels retain their independent display framing and explicit labels.
"""
from pathlib import Path
import json


def compose(before,after,output,*,labels=('Before: physics','After: physics')):
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    from real2sim2real.artifacts import digest,write_json
    folders=list(map(Path,(before,after)));output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous comparisons')
    if len(labels)!=2 or any(not isinstance(x,str) or not x.strip() for x in labels):
        raise ValueError('Two explicit physical-panel labels required')
    proofs=[json.loads((p/'video_verification.json').read_text()) for p in folders]
    for folder,proof in zip(folders,proofs):
        if (proof['comparison']!='reference_vs_physics' or proof['size']!=[1280,640]
                or digest(folder/'reference_vs_physics.mp4')!=proof['sha256']):
            raise ValueError('Require a verified reference-versus-physics input')
    if proofs[0]['frames']!=proofs[1]['frames'] or abs(proofs[0]['fps']-proofs[1]['fps'])>1e-6:
        raise ValueError('Physical panels have different clocks or frame counts')
    output.mkdir(parents=True);readers=[];writer=None;count=0
    try:
        for folder in folders:
            reader=imageio_ffmpeg.read_frames(str(folder/'reference_vs_physics.mp4'));next(reader);readers.append(reader)
        writer=imageio_ffmpeg.write_frames(str(output/'comparison.partial.mp4'),(1280,640),fps=proofs[0]['fps'],
            codec='libx264',quality=7,macro_block_size=1,output_params=['-preset','veryfast','-threads','1','-movflags','+faststart']);writer.send(None)
        font=ImageFont.load_default(size=19)
        for frame in range(proofs[0]['frames']):
            canvas=Image.new('RGB',(1280,640),'white')
            for column,reader in enumerate(readers):
                raw=next(reader);panel=Image.frombytes('RGB',(1280,640),raw).crop((640,0,1280,640))
                canvas.paste(panel,(column*640,0))
            draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,1280,70),fill='white')
            for column,label in enumerate(labels):draw.text((15+column*640,20),label,font=font,fill=(23,32,51))
            draw.rectangle((0,585,1280,640),fill='white')
            draw.text((15,590),f'Time {frame/proofs[0]["fps"]:.2f}s | independently framed displays | saved physical states only',font=font,fill=(23,32,51))
            writer.send(np.asarray(canvas));count+=1
        for reader in readers:
            if next(reader,None) is not None:raise ValueError('Input contains unexpected extra frames')
    finally:
        for reader in readers:reader.close()
        if writer is not None:writer.close()
    target=output/'comparison.mp4';(output/'comparison.partial.mp4').replace(target)
    decoder=imageio_ffmpeg.read_frames(str(target));next(decoder);decoded=sum(1 for _ in decoder);decoder.close()
    if decoded!=count:raise ValueError('Composite decode frame count mismatch')
    proof=dict(frames=decoded,fps=proofs[0]['fps'],sha256=digest(target),labels=list(labels),
        source_videos=[p['sha256'] for p in proofs],physical_only=True,same_camera=False,
        display_framing='Independent original render framing retained',simulation_rerun=False,actor_training=False)
    write_json(output/'video_verification.json',proof);return target
