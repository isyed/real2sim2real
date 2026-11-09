"""Package a portable frozen SONIC plus fixed-hand controller, without an actor.

Weights and robot assets are supplied locally and copied or hard-linked into the
output artifact. Packaging imports no simulator and does not launch training.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import os
import shutil
import xml.etree.ElementTree as ET

from real2sim2real.artifacts import load, publish, digest, write_json


def validate_config(config):
    action=config['action'];env=config['env'];task=config['task'];scene=config['scene']
    if (action['_target_']!='flash_chord.runtime.actions.SonicJointResidualActionConfig'
            or not action.get('hand_closing') or env['action']!=action):
        raise ValueError('Require matching frozen SONIC and fixed-hand actions')
    if (env['voc_scale']!=0 or config['voc_scale_override']!=0 or
            any(env['voc'][name]!=0 for name in ['max_force','max_torque'])):
        raise ValueError('Frozen baseline must disable object assistance')
    if (scene['world_count']!=1 or not isinstance(task.get('source_frame_playback'),bool)
            or task.get('motion_start_frame')!=0 or task.get('motion_speed')!=1.):
        raise ValueError('Require one-world, frame-zero, unchanged-speed evaluation')
    if (config['training'].get('resume') or config['training'].get('checkpoint_path')
            or config['training'].get('curriculum') is not None):
        raise ValueError('Frozen baseline cannot resume an actor or assistance curriculum')
    if config['training'].get('_target_')!='flash_chord.training.flash_sac.config.TrainingConfig':
        raise ValueError('Frozen evaluation requires the injectable FlashSAC policy seam')


def configuration():
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    with initialize_config_module(version_base='1.3',config_module='flash_chord.configs'):
        cfg=compose(config_name='train',overrides=[
            'experiment=g1_bounded_flash_sac','task.parquet=reference/reference.parquet',
            'logging.mode=disabled','scene.world_count=1'])
    config=OmegaConf.to_container(cfg,resolve=True)
    config['action']['policy_dir']='assets/policies/sonic'
    config['task'].update(motion_start_frame=0,motion_speed=1.,source_frame_playback=True)
    config['env'].update(voc_scale=0.,auto_reset=False)
    config['env']['voc'].update(max_force=0.,max_torque=0.)
    config['training'].update(resume=False,checkpoint_path=None,curriculum=None)
    config['voc_scale_override']=0.
    config['env']['action']=deepcopy(config['action'])
    validate_config(config)
    return config


def copy_tree(source,target,*,hardlink):
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Portable bundles cannot depend on symlinks')
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        destination=target/path.relative_to(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if hardlink:
            os.link(path,destination)
        else:
            shutil.copy2(path,destination)


def copy_collision_cache(source,target,*,hardlink=False):
    """Pin explicitly supplied content-keyed CoACD results without recomputation.

    Cache names remain the runtime's geometry/parameter keys. The caller owns
    their provenance; array validation and artifact hashes make reuse auditable.
    """
    import re
    import numpy as np
    source=Path(source)
    if not source.is_dir() or source.is_symlink():raise ValueError('Expected a real collision-cache directory')
    names=[]
    for path in sorted(source.glob('*.npz')):
        if not re.fullmatch(r'[0-9a-f]{16}\.npz',path.name):continue
        if path.is_symlink():raise ValueError('Collision caches cannot contain symlinks')
        with np.load(path,allow_pickle=False) as data:
            n=np.asarray(data['n'])
            if n.shape!=() or not np.issubdtype(n.dtype,np.integer) or int(n)<1:
                raise ValueError('Invalid collision-cache hull count')
            count=int(n)
            if set(data.files)!={'n',*[f'{key}{i}' for i in range(count) for key in ('v','f')]}:
                raise ValueError('Unexpected collision-cache members')
            for i in range(count):
                vertices,faces=np.asarray(data[f'v{i}']),np.asarray(data[f'f{i}'])
                if (vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all()
                        or faces.ndim!=2 or faces.shape[1]!=3 or not len(faces)
                        or not np.issubdtype(faces.dtype,np.integer) or faces.min()<0 or faces.max()>=len(vertices)):
                    raise ValueError('Invalid collision-cache geometry')
        target.mkdir(parents=True,exist_ok=True)
        if hardlink:os.link(path,target/path.name)
        else:shutil.copy2(path,target/path.name)
        names.append(path.name)
    if not names:raise ValueError('No complete content-keyed collision caches found')
    return names


def package(reference,assets,output,*,hardlink=False,collision_cache=None):
    reference,assets,output=map(lambda x:Path(x).resolve(),[reference,assets,output])
    record=load(reference,kinds={'robot-reference'})
    if record['metadata'].get('robot')!='g1':
        raise ValueError('This recipe requires the G1/Dex3 embodiment')
    if output.exists() or output.is_relative_to(reference) or output.is_relative_to(assets):
        raise ValueError('Use a fresh output outside the input asset/reference trees')
    required=['policies/sonic/encoder_batched.onnx','policies/sonic/decoder_batched.onnx',
              'urdfs/g1/main_with_hand.urdf']
    for name in required:
        if not (assets/name).is_file():
            raise FileNotFoundError(assets/name)
    config=configuration()
    config['scene']['support']='support' in record['files']
    # Preserve source duration when camera and controller clocks differ.
    source_fps=record['metadata'].get('fps')
    if source_fps is not None:
        import math
        if not math.isfinite(source_fps) or source_fps<=0:raise ValueError('Invalid source FPS')
        config['task']['source_frame_playback']=math.isclose(source_fps,config['env']['sim']['fps'],rel_tol=1e-8)
    output.mkdir(parents=True)
    copy_tree(reference,output/'reference',hardlink=hardlink)
    for name in ['policies/sonic','urdfs/g1','meshes/g1']:
        if not (assets/name).exists() and name=='meshes/g1':
            continue
        copy_tree(assets/name,output/'assets'/name,hardlink=hardlink)
    for urdf in output.rglob('*.urdf'):
        for mesh in ET.parse(urdf).getroot().iter('mesh'):
            path=(urdf.parent/mesh.attrib['filename']).resolve()
            if not path.is_relative_to(output) or not path.is_file():
                raise ValueError('URDF mesh is missing or outside the portable bundle: '+str(path))
    cached=copy_collision_cache(collision_cache,output/'assets/.cache/convex',hardlink=hardlink) if collision_cache is not None else []
    native_reference='reference/'+record['files']['reference']['path']
    playback_reference=native_reference
    if not config['task']['source_frame_playback']:
        os.environ['R2S2R_ASSETS']=str(output/'assets')
        from .reference_timing import prepare
        playback_reference='reference/control_reference.parquet'
        prepare(output/native_reference,output/playback_reference,control_fps=config['env']['sim']['fps'])
        config['task']['source_frame_playback']=True
    manifest=dict(schema='real2sim2real.frozen-controller/1',controller='frozen_sonic_fixed_hands',
        config=config,reference=playback_reference,native_reference=native_reference,
        playback_fps=config['env']['sim']['fps'],
        reference_artifact_sha256=digest(reference/'artifact.json'),
        source_fps=record['metadata'].get('fps'),
        actor_present=False,training_steps=0,correction='deterministic_zeros',
        object_assistance_scale=0.,physical_deployment_approved=False,
        collision_cache_files=cached)
    manifest['config']['task']['parquet']=manifest['reference']
    write_json(output/'manifest.json',manifest)
    files={'manifest':output/'manifest.json'}
    files.update({f'file_{i}':p for i,p in enumerate(sorted(output.rglob('*')))
                  if p.is_file() and p!=output/'manifest.json'})
    publish(output,'frozen-controller',files,metadata=dict(robot='g1',actor_present=False,
        training_steps=0,physics_validated=False,reference_artifact_sha256=manifest['reference_artifact_sha256']))
    return output


def materialize(policy):
    policy=Path(policy).resolve()
    load(policy,kinds={'frozen-controller'})
    manifest=json.loads((policy/'manifest.json').read_text())
    if (manifest.get('schema')!='real2sim2real.frozen-controller/1'
            or manifest.get('actor_present') is not False or manifest.get('training_steps')!=0
            or manifest.get('correction')!='deterministic_zeros'):
        raise ValueError('Not a frozen zero-correction controller')
    config=deepcopy(manifest['config']);validate_config(config)
    if manifest.get('source_fps') is not None and config['task']['source_frame_playback']:
        import math
        if not math.isclose(manifest.get('playback_fps',manifest['source_fps']),config['env']['sim']['fps'],rel_tol=1e-8):
            raise ValueError('Source-frame playback would change motion speed; repackage with time resampling')
    reference=(policy/manifest['reference']).resolve()
    if not reference.is_relative_to(policy):
        raise ValueError('Reference escaped its policy bundle')
    config['task']['parquet']=str(reference)
    config['action']['policy_dir']=str(policy/'assets/policies/sonic')
    config['env']['action']=deepcopy(config['action'])
    return manifest,config


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','assets','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hardlink',action='store_true',help='Same-filesystem bundles; no symlinks')
    parser.add_argument('--collision-cache',type=Path,help='Optional provenance-checked CoACD cache directory to pin in the bundle')
    args=parser.parse_args()
    print(package(args.reference,args.assets,args.output,hardlink=args.hardlink,collision_cache=args.collision_cache))
