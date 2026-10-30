"""CPU model forwards and portable asset verification before frozen simulation.

This does not execute physics or establish task success. Heavy runtime imports
are lazy; explicitly hide CUDA before calling this CPU-only preflight.
"""
from pathlib import Path
import os
from real2sim2real.artifacts import digest


def verify(policy, *, expected_frames):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Hide CUDA for CPU-only frozen-controller verification')
    import numpy as np
    import onnxruntime as ort
    from real2sim2real.controller_actor_training.frozen import materialize
    policy=Path(policy).resolve()
    manifest,cfg=materialize(policy)
    os.environ['R2S2R_ASSETS']=str(policy/'assets')
    from flash_chord.data import load_reference
    native=load_reference(policy/manifest.get('native_reference',manifest['reference']))
    mode=cfg['task']['source_frame_playback'];control_fps=cfg['env']['sim']['fps']
    ref=load_reference(cfg['task']['parquet'],control_fps=control_fps,motion_speed=1.,source_frame_playback=mode)
    if native.num_frames!=expected_frames or ref.robot_joint_pos().shape!=(ref.num_frames,43):raise ValueError('Wrong reference coverage')
    names=ref.robot_joint_names();finger=[i for i,n in enumerate(names) if '_hand_' in n]
    if len(finger)!=14 or np.any(ref.robot_joint_pos()[:,finger]!=0):raise ValueError('Unexpected finger reference')
    for asset in ref.object_assets():
        if not Path(asset.urdf_path).is_file():
            raise ValueError('Missing portable object asset')
    import xml.etree.ElementTree as ET
    for urdf in list((policy/'reference').rglob('*.urdf'))+[policy/'assets/urdfs/g1/main_with_hand.urdf']:
        for mesh in ET.parse(urdf).getroot().iter('mesh'):
            path=(urdf.parent/mesh.attrib['filename']).resolve()
            if not path.is_relative_to(policy) or not path.is_file():
                raise ValueError('URDF asset is missing or outside the portable policy: '+str(path))
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    models=[]
    for name in ['encoder_batched.onnx','decoder_batched.onnx']:
        path=policy/'assets/policies/sonic'/name
        model=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        feed={i.name:np.zeros([n if isinstance(n,int) else 1 for n in i.shape],np.float32) for i in model.get_inputs()}
        output=model.run(None,feed)
        if not all(np.isfinite(v).all() for v in output):raise ValueError('Nonfinite SONIC CPU forward')
        models.append(dict(model=name,sha256=digest(path),output_shapes=[list(v.shape) for v in output]))
    return dict(passed=True,cpu_only=True,frames=ref.num_frames,source_frames=native.num_frames,control_frames=ref.num_frames,source_fps=native.fps,
        playback_fps=ref.fps,
        control_fps=control_fps,source_frame_playback=mode,models=models,policy_sha256=digest(policy/'manifest.json'),
        reference_sha256=digest(Path(cfg['task']['parquet'])),actor_present=False,
        geometric_contact_frames={side:int(np.count_nonzero(ref.contact_active(side))) for side in ref.sides})
