"""Expose propagated subject masks on RGB before expensive geometry inference.

A nonempty, in-bounds mask can select background or the wrong subject. This
review deliberately records semantics as unverified; area alone is no approval.
"""
from pathlib import Path
import json
import numpy as np
from real2sim2real.artifacts import digest,write_json


def sample_overlay(rgb,masks,roles):
    from PIL import Image,ImageDraw
    image=np.asarray(rgb)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('Expected native uint8 RGB')
    if len(masks)!=len(roles) or not masks:raise ValueError('Each mask requires a role')
    h,w=image.shape[:2];width=512;height=max(1,round(h*width/w));rows=[]
    canvas=Image.new('RGB',(width*(len(masks)+1),height+32),'#15202b')
    canvas.paste(Image.fromarray(image).resize((width,height)),(0,32));draw=ImageDraw.Draw(canvas)
    draw.text((8,8),'Source RGB',fill='white')
    for j,(mask,role) in enumerate(zip(masks,roles)):
        m=np.asarray(mask)
        if m.shape!=(h,w):raise ValueError('Mask and source RGB use different pixel grids')
        if not np.isfinite(m).all():raise ValueError('Nonfinite mask')
        m=m!=0;y,x=np.nonzero(m);color=np.array([(235,55,55),(245,180,35),(45,210,225)][j%3])
        pixels=image.copy();pixels[m]=(pixels[m]*.4+color*.6).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels).resize((width,height)),((j+1)*width,32))
        draw.text(((j+1)*width+8,8),str(role)+' — verify identity',fill='white')
        rows.append(dict(role=str(role),foreground_pixels=int(m.sum()),
            bbox_xyxy=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None))
    return canvas,rows


def review_masks(video,masks_root,output,*,frame_count,source_video_sha256,prompts,frame_ids=None):
    from PIL import Image
    import av
    if isinstance(frame_count,bool) or not isinstance(frame_count,int) or frame_count<1:
        raise ValueError('Expected positive source frame count')
    if digest(video)!=source_video_sha256:raise ValueError('Review source video changed')
    ids=sorted(set(frame_ids if frame_ids is not None else np.linspace(0,frame_count-1,min(5,frame_count)).round().astype(int).tolist()))
    if not ids or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<frame_count for i in ids):
        raise ValueError('Review frames must be source timeline indices')
    masks_root,output=Path(masks_root),Path(output);roles=['human','object'];samples=[]
    for identity in (0,1):
        actual=sorted(x.name for x in (masks_root/str(identity)).glob('*.png'))
        expected=[f'{i:06d}.png' for i in range(frame_count)]
        if actual!=expected:raise ValueError('Segmentation masks do not cover the complete source timeline')
    output.mkdir(parents=True,exist_ok=True);previews=[];decoded=0
    with av.open(str(video)) as source:
        for i,frame in enumerate(source.decode(video=0)):
            decoded+=1
            if i not in ids:continue
            paths=[masks_root/str(j)/f'{i:06d}.png' for j in (0,1)]
            arrays=[]
            for path in paths:
                with Image.open(path) as mask:arrays.append(np.asarray(mask.convert('L')))
            preview,stats=sample_overlay(frame.to_ndarray(format='rgb24'),arrays,roles)
            path=output/f'frame_{i:06d}.jpg';preview.save(path,quality=90);previews.append(preview)
            samples.append(dict(frame_index=i,masks=stats,mask_sha256=[digest(p) for p in paths],
                preview=path.name,preview_sha256=digest(path)))
    if decoded!=frame_count or len(samples)!=len(ids):raise ValueError('Review RGB timeline differs from masks')
    sheet=Image.new('RGB',(previews[0].width,sum(p.height for p in previews)));y=0
    for p in previews:sheet.paste(p,(0,y));y+=p.height
    sheet.save(output/'contact_sheet.jpg',quality=90)
    result=dict(schema='real2sim2real.mask-review/1',source_video_sha256=source_video_sha256,
        prompts_sha256=digest(prompts),source_frames=frame_count,mask_timelines_complete=True,
        reviewed_frame_indices=ids,samples=samples,semantics_verified=False,
        empty_sample_masks=[dict(frame_index=s['frame_index'],role=m['role']) for s in samples for m in s['masks'] if not m['foreground_pixels']],
        limitations=['Inspect which subject each mask selects; foreground counts alone cannot verify human or object identity.',
            'Sampled previews do not certify all-frame mask quality or physical correctness.'])
    write_json(output/'mask_review.json',result);return result
