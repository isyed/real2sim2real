"""Isolated settling bound to the exact frozen policy pose/support/settings."""
import argparse,os
from pathlib import Path
from real2sim2real.artifacts import digest,write_json


def run(policy,output,*,device='cpu',seconds=3.):
    if device=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for a CPU probe')
    import numpy as np
    from .frozen import materialize
    from .settling_compatibility import isolated_run_parameters,compare_solver_settings
    from .isolated_object_stability import run as probe,assess_settling
    from flash_chord.data import load_reference
    from flash_chord.assets.registry import support_usda_for_reference
    policy,output=Path(policy).resolve(),Path(output).resolve()
    manifest,cfg=materialize(policy);os.environ['R2S2R_ASSETS']=str(policy/'assets')
    ref=load_reference(cfg['task']['parquet'],control_fps=cfg['sim']['fps'],motion_speed=1.,source_frame_playback=cfg['task']['source_frame_playback'])
    assets=ref.object_assets()
    if len(assets)!=1:raise ValueError('Isolated policy probe supports one rigid object')
    urdf=Path(assets[0].urdf_path).resolve()
    if not urdf.is_relative_to(policy):raise ValueError('Object asset escapes portable policy')
    support=support_usda_for_reference(ref) if cfg['scene']['support'] else None
    if cfg['scene']['support'] and support is None:raise ValueError('Declared support missing')
    if support is not None and not Path(support).resolve().is_relative_to(policy):raise ValueError('Support escapes portable policy')
    parameters=isolated_run_parameters(cfg['sim'])
    scene={key:cfg['scene'][key] for key in ['contact_friction','object_free_joint_damping']}
    pose=np.r_[ref.object_body_pos_w()[0,0],ref.object_body_quat_w()[0,0][[1,2,3,0]]]
    result=probe(urdf,pose,output,device=device,seconds=seconds,friction=scene['contact_friction'],
        support_usda=support,scene_parameters=scene,**parameters)
    compatibility=compare_solver_settings(result,cfg['sim']);gate=assess_settling(result)
    if not compatibility['recorded_solver_settings_match'] or result['matched_scene_parameters']!=scene:
        raise ValueError('Isolated probe settings differ from policy')
    receipt=dict(policy_sha256=digest(policy/'manifest.json'),urdf_sha256=digest(urdf),
        pose_xyzw=pose.tolist(),support_sha256=digest(support) if support is not None else None,
        solver_compatibility=compatibility,scene_parameters=scene,gate=gate,
        backend=result['backend'],physical_task_success_established=False)
    write_json(output/'POLICY_ADMISSION.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda:0'],default='cpu');p.add_argument('--seconds',type=float,default=3.)
    args=p.parse_args();run(args.policy,args.output,device=args.device,seconds=args.seconds)
