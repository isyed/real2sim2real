"""CPU-only RGB evidence at foreground-depth warnings; never a repaired pose."""
from pathlib import Path
import json,math
import numpy as np


def warning_frames(report,fps):
    """Show onset and later context; single-frame filters can miss sustained bias."""
    n=report['source_frames'];warnings=report['temporal_depth_warning_frames']
    if type(n) is not int or n<1 or not np.isfinite(fps) or fps<=0:
        raise ValueError('Positive source coverage and frame rate required')
    if any(type(i) is not int or not 0<=i<n for i in warnings):raise ValueError('Source-indexed warnings required')
    first=min(warnings) if warnings else 0
    offsets=[-1,0,round(fps/6),round(fps),round(5*fps/3),round(3*fps)]
    return sorted({max(0,min(n-1,first+i)) for i in offsets})


def render(source,video,report_file,output,*,frames=None):
    """Pin RGB/report/artifact hashes and decode the complete source timeline.

    Selects at most six source frames. Labels distinguish raw and aligned depth;
    a visible depth patch and an attractive crop do not establish metric truth.
    """
    import h5py,imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageOps
    from real2sim2real.artifacts import load,digest,role_path,write_json
    source,video,report_file,output=map(Path,(source,video,report_file,output))
    if output.exists():raise FileExistsError('Preserve existing depth preview')
    artifact=load(source,kinds={'reconstruction'});report=json.loads(report_file.read_text())
    if (report['schema']!='real2sim2real.foreground-visibility-audit/1'
        or report['source_identity']['artifact_sha256']!=digest(source/'artifact.json')
        or artifact['metadata'].get('source_video_sha256')!=digest(video)):
        raise ValueError('Depth audit and RGB must match this reconstruction')
    if len(report['rows'])!=report['source_frames']:raise ValueError('Complete audit timeline required')
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader)
    selected=warning_frames(report,meta['fps']) if frames is None else list(frames)
    if (not selected or len(selected)>6 or sorted(set(selected))!=selected
        or any(type(i) is not int or not 0<=i<report['source_frames'] for i in selected)):
        reader.close();raise ValueError('One to six unique ordered source frame IDs required')
    width,height=meta['size'];tiles={};count=0;measurements=[]
    try:
        with h5py.File(role_path(source,'object_masks')) as masks:
            if masks['frames'].shape!=(report['source_frames'],height,width):raise ValueError('Native mask/video grids differ')
            for count,encoded in enumerate(reader,1):
                i=count-1
                if i not in selected:continue
                rgb=Image.frombytes('RGB',(width,height),encoded);mask=masks['frames'][i]!=0
                y,x=np.where(mask);box=(0,0,width,height)
                if len(x):
                    margin=max(32,round(max(np.ptp(x),np.ptp(y))*.55))
                    box=(max(0,int(x.min())-margin),max(0,int(y.min())-margin),min(width,int(x.max())+margin+1),min(height,int(y.max())+margin+1))
                crop=ImageOps.contain(rgb.crop(box),(300,330));tile=Image.new('RGB',(300,385),'#18212c')
                tile.paste(crop,((300-crop.width)//2,55+(330-crop.height)//2));draw=ImageDraw.Draw(tile)
                aligned=report['rows'][i]['median_depth_m'];raw=None
                if 'depth_origin'in report:raw=report['depth_origin']['raw']['rows'][i]['median_depth_m']
                def depth(value):return 'unavailable' if value is None else f'{value:.2f} m'
                draw.text((8,5),f'Frame {i} | aligned prediction {depth(aligned)}',fill='white')
                draw.text((8,23),f'Raw prediction {depth(raw)}',fill='white')
                draw.text((8,40),'RGB evidence; true depth unverified',fill='#f0cd8c')
                tiles[i]=tile;measurements.append(dict(frame=i,raw_depth_m=raw,aligned_depth_m=aligned,crop_xyxy=list(box)))
        if count!=report['source_frames'] or len(tiles)!=len(selected):raise ValueError('Decoded RGB/audit coverage differs')
    finally:reader.close()
    output.mkdir(parents=True);canvas=Image.new('RGB',(900,385*math.ceil(len(selected)/3)),'#18212c')
    for j,i in enumerate(selected):canvas.paste(tiles[i],((j%3)*300,(j//3)*385))
    image=output/'depth_warning_rgb.jpg';canvas.save(image,quality=92)
    receipt=dict(schema='real2sim2real.depth-warning-preview/1',source_frames=count,frames=selected,measurements=measurements,
        artifact_sha256=digest(source/'artifact.json'),video_sha256=digest(video),report_sha256=digest(report_file),
        image_sha256=digest(image),depth_modified=False,poses_modified=False,metric_truth_verified=False,
        selection='Explicit source frames' if frames is not None else 'First warning and temporal context',
        limitation='Spatially coherent or temporally stable predictions may still have biased foreground depth; inspect RGB and independent evidence.')
    write_json(output/'preview.json',receipt);return receipt


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','video','report','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(render(a.input,a.video,a.report,a.output)))
