"""Explicit frame-zero rest corrections without changing later task motion.

Only a single rigid, unarticulated object with inactive initial hand contacts
is supported. The caller must supply source/settling evidence. Robot targets,
human motion, geometry, contact labels and all later poses stay unchanged.
This is a hypothesis package; native image and GPU physics checks still apply.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def revise_row(original, pose_xyzw, *, max_translation_m=.03,
               max_turn_deg=8., max_seam_m=.08, max_seam_deg=30.):
    p=np.asarray(pose_xyzw,float);xyz=np.asarray(original['object_body_position'],float)
    wxyz=np.asarray(original['object_body_wxyz'],float);contacts=np.asarray(original['hand_contact_active'])
    limits=np.asarray([max_translation_m,max_turn_deg,max_seam_m,max_seam_deg],float)
    if (p.shape!=(7,) or not np.isfinite(p).all() or not np.isclose(np.linalg.norm(p[3:]),1.,atol=1e-6)
        or xyz.ndim!=3 or xyz.shape[1:]!=(1,3) or len(xyz)<2 or wxyz.shape!=(len(xyz),1,4)
        or contacts.shape!=(2,len(xyz)) or not np.isfinite(contacts).all() or np.any(contacts[:,0]!=0)
        or np.any(np.asarray(original['object_articulation'])!=0)
        or not np.isfinite(limits).all() or np.any(limits<=0)):
        raise ValueError('Bounded rigid single-object rest pose with inactive initial contacts required')
    if (not np.isfinite(xyz).all() or not np.isfinite(wxyz).all()
        or not np.allclose(np.linalg.norm(wxyz,axis=-1),1.,atol=1e-6)
        or not np.allclose(np.asarray(original['object_root_position']),xyz[:,0],atol=1e-7)):
        raise ValueError('Finite unit rigid poses and matching root/body positions required')
    rotations=Rotation.from_quat(wxyz[:,0][:,[1,2,3,0]])
    root=Rotation.from_rotvec(np.asarray(original['object_root_axis_angle']))
    if not np.allclose(root.as_matrix(),rotations.as_matrix(),atol=2e-6):
        raise ValueError('Root/body rotation representations disagree')
    proposed=Rotation.from_quat(p[3:])
    change=float(np.linalg.norm(p[:3]-xyz[0,0]))
    turn=float(np.rad2deg((proposed*rotations[0].inv()).magnitude()))
    seam=float(np.linalg.norm(p[:3]-xyz[1,0]))
    seam_turn=float(np.rad2deg((rotations[1]*proposed.inv()).magnitude()))
    if change>max_translation_m or turn>max_turn_deg or seam>max_seam_m or seam_turn>max_seam_deg:
        raise ValueError('Correction or first native-frame seam exceeds declared bound')
    row=deepcopy(original)
    row['object_body_position'][0]=[p[:3].tolist()]
    row['object_body_wxyz'][0]=[p[[6,3,4,5]].tolist()]
    row['object_root_position'][0]=p[:3].tolist()
    row['object_root_axis_angle'][0]=proposed.as_rotvec().tolist()
    return row,dict(changed_native_frame_ids=[0],translation_change_m=change,rotation_change_deg=turn,
        first_native_seam_m=seam,first_native_seam_deg=seam_turn,later_frames_unchanged=True,
        robot_targets_unchanged=True,hand_contacts_unchanged=True,mesh_and_inertia_unchanged=True,
        limits=dict(translation_m=max_translation_m,rotation_deg=max_turn_deg,seam_m=max_seam_m,seam_deg=max_seam_deg))


def package(source, output, pose_xyzw, *, evidence, hardlink=False, **limits):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from real2sim2real.artifacts import load,digest,publish,write_json
    from .frozen import copy_tree
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output.is_relative_to(source) or not isinstance(evidence,dict) or not evidence:
        raise ValueError('Fresh independent output and explicit source evidence required')
    artifact=load(source,kinds={'robot-reference'})
    table=pq.read_table(source/artifact['files']['reference']['path'])
    if table.num_rows!=1:raise ValueError('One native motion sequence required')
    row,receipt=revise_row(table.to_pylist()[0],pose_xyzw,**limits)
    before=digest(source/'artifact.json');copy_tree(source,output,hardlink=hardlink)
    target=output/artifact['files']['reference']['path'];target.unlink()
    pq.write_table(pa.Table.from_pylist([row],schema=table.schema),target)
    receipt.update(schema='real2sim2real.initial-rest-reference/1',evidence=evidence,
        parent_artifact_sha256=before,physical_execution_validated=False,training_approved=False)
    write_json(output/'initial_rest_revision.json',receipt);(output/'artifact.json').unlink()
    files={role:output/member['path'] for role,member in artifact['files'].items()}
    files['initial_rest_revision']=output/'initial_rest_revision.json'
    publish(output,'robot-reference',files,metadata=dict(artifact['metadata'],parent_artifact_sha256=before,
        physics_validated=False,initial_rest_revision=True))
    load(source,kinds={'robot-reference'})
    if digest(source/'artifact.json')!=before:raise ValueError('Parent changed')
    return receipt
