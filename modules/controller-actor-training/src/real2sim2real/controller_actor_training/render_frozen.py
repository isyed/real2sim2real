"""CPU-only matched video from saved kinematic targets and physical body poses.

Display meshes may be simplified; source robot/reference assets and simulation
geometry are never changed. This module does not run physics or a controller.
"""
from pathlib import Path
import argparse
import json
import os
import xml.etree.ElementTree as ET


def visual_meshes(urdf,labels,prefix,*,object_mesh=False):
    import numpy as np
    from scipy.spatial.transform import Rotation
    import trimesh
    pieces=[]
    for link in ET.parse(urdf).getroot().findall('link'):
        visuals=link.findall('visual')
        if not visuals:continue
        name=prefix+link.attrib['name']
        if name not in labels:raise ValueError('Visual body missing from recorded model: '+name)
        for visual in visuals:
            source=visual.find('geometry/mesh')
            if source is None:raise ValueError('This CPU renderer currently requires mesh visuals')
            mesh=trimesh.load((urdf.parent/source.attrib['filename']).resolve(),force='mesh',process=False)
            mesh.merge_vertices(merge_tex=True,merge_norm=True)
            if not object_mesh:mesh=mesh.convex_hull
            budget=3000 if object_mesh else 120
            if len(mesh.faces)>budget:mesh=mesh.simplify_quadric_decimation(face_count=budget)
            origin=visual.find('origin')
            xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            scale=np.fromstring(source.get('scale','1 1 1'),sep=' ')
            vertices=(np.asarray(mesh.vertices)*scale)@Rotation.from_euler('xyz',rpy).as_matrix().T+xyz
            color=(239,133,49) if object_mesh else ((61,143,179) if 'left_' in name else (110,119,186))
            pieces.append((labels.index(name),vertices[mesh.faces],color))
    return pieces


def render(policy,rollout,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide GPU for CPU rendering')
    import numpy as np
    from scipy.spatial.transform import Rotation
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    from real2sim2real.artifacts import digest,write_json
    from .frozen import materialize
    policy,rollout,output=map(lambda x:Path(x).resolve(),[policy,rollout,output])
    if output.exists():raise FileExistsError('Preserve completed videos')
    manifest,config=materialize(policy);os.environ['R2S2R_ASSETS']=str(policy/'assets')
    from flash_chord.data import load_reference
    reference=load_reference(config['task']['parquet'],control_fps=50.,motion_speed=1.,source_frame_playback=True)
    if config['scene']['support']:
        raise ValueError('This renderer currently supports ground-only scenes; add support rendering explicitly')
    audit=json.loads((rollout/'audit.json').read_text());labels=audit['body_labels']
    records=[]
    for name,label in [('kinematic','Retargeted reference (no physics)'),('physical','Frozen SONIC + fixed hands (physics)')]:
        with np.load(rollout/(name+'_states.npz')) as z:poses=z['body_q'].copy()
        if poses.shape!=(audit['source_frames'],len(labels),7) or not np.isfinite(poses).all():
            raise ValueError('Incomplete or incompatible recorded states')
        records.append(dict(label=label,poses=poses))
    pieces=visual_meshes(policy/'assets/urdfs/g1/main_with_hand.urdf',labels,'g1/')
    for asset in reference.object_assets():
        urdf=Path(asset.urdf_path);prefix=ET.parse(urdf).getroot().attrib['name']+'/'
        pieces+=visual_meshes(urdf,labels,prefix,object_mesh=True)
    ids=np.concatenate([np.full(len(tri),body,int) for body,tri,_ in pieces])
    local=np.concatenate([tri for _,tri,_ in pieces])
    palette=np.concatenate([np.tile(color,(len(tri),1)) for _,tri,color in pieces])
    az,el=np.deg2rad([-58.,18.]);view=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
    right=np.array([-np.sin(az),np.cos(az),0.]);camera=np.stack([right,np.cross(view,right),view])
    def world(record,frame):
        pose=record['poses'][frame];rot=Rotation.from_quat(pose[:,3:]).as_matrix()
        return np.einsum('fij,fvj->fvi',rot[ids],local)+pose[ids,None,:3]
    count=audit['source_frames'];fps=audit['control_fps']
    sample=np.concatenate([world(r,int(i)).reshape(-1,3) for r in records for i in np.linspace(0,count-1,20)])@camera.T
    lo,hi=sample[:,:2].min(0),sample[:,:2].max(0);center=(lo+hi)*.5
    scale=min(580/max(hi[0]-lo[0],1.),470/max(hi[1]-lo[1],1.))*.94
    output.mkdir(parents=True)
    font=ImageFont.load_default(size=18)
    writer=imageio_ffmpeg.write_frames(str(output/'reference_vs_physics.partial.mp4'),(1280,640),fps=fps,
        codec='libx264',quality=7,macro_block_size=1,
        output_params=['-preset','veryfast','-threads','1','-movflags','+faststart']);writer.send(None)
    light=np.array([.3,-.4,.86]);light/=np.linalg.norm(light)
    try:
        for frame in range(count):
            canvas=Image.new('RGB',(1280,640),(248,250,252));draw=ImageDraw.Draw(canvas)
            for col,record in enumerate(records):
                panel=Image.new('RGB',(620,500),(240,244,248));d=ImageDraw.Draw(panel)
                for value in range(-5,6):
                    for line in [np.array([[value,-5,0],[value,5,0]]),np.array([[-5,value,0],[5,value,0]])]:
                        xy=((line@camera.T)[:,:2]-center)*[scale,-scale]+[310,250]
                        d.line([tuple(p) for p in xy],fill=(216,222,229))
                tri=world(record,frame);normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
                colors=np.clip(palette*(.5+.5*np.abs(normal@light))[:,None],0,255).astype(np.uint8)
                projected=tri@camera.T;xy=(projected[:,:,:2]-center)*[scale,-scale]+[310,250]
                polygons=xy.reshape(-1,6).tolist();rgb=[tuple(c) for c in colors.tolist()]
                for i in np.argsort(projected[:,:,2].mean(1)):d.polygon(polygons[i],fill=rgb[i])
                canvas.paste(panel,(10+640*col,85));draw.text((20+640*col,20),record['label'],font=font,fill=(23,32,51))
            draw.text((20,590),f'Frame {frame}/{count-1} | control time {frame/fps:.2f}s | same camera and scale',font=font,fill=(23,32,51))
            draw.text((20,615),'No actor training or object assistance. Simplified display meshes; saved states only.',font=font,fill=(71,85,105))
            writer.send(np.asarray(canvas))
            if frame%100==0:write_json(output/'progress.json',dict(rendered=frame+1,total=count))
    finally:writer.close()
    video=output/'reference_vs_physics.mp4';(output/'reference_vs_physics.partial.mp4').replace(video)
    decoder=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(decoder)
    decoded=sum(1 for _ in decoder);decoder.close()
    if decoded!=count:raise ValueError('Video frame count differs from recorded states')
    write_json(output/'video_verification.json',dict(frames=decoded,fps=fps,size=meta['size'],sha256=digest(video),
        same_camera=True,no_physics_rerun=True,no_actor_training=True,
        reference_states_sha256=digest(rollout/'kinematic_states.npz'),physical_states_sha256=digest(rollout/'physical_states.npz'),
        policy_manifest_sha256=digest(policy/'manifest.json')))
    return video


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['policy','rollout','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(render(args.policy,args.rollout,args.output))
