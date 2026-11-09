"""Explicit native-frame windows for diagnostic controller experiments.

Never infer a task phase or mutate the full reference. Windows carry their
native frame range and do not constitute a full-task deployment evaluation.
"""
from copy import deepcopy


def slice_row(original, start, stop):
    frames=len(original['robot_root_position'])
    if type(start) is not int or type(stop) is not int or not 0<=start<stop<=frames or stop-start<2:
        raise ValueError('Explicit native half-open window of at least two frames required')
    row=deepcopy(original)
    temporal=('robot_root_position','robot_root_wxyz','robot_joint_positions','ee_pose_w',
        'object_body_position','object_body_wxyz','object_root_position','object_root_axis_angle',
        'object_articulation','ik_error_per_frame','ik_num_iterations','frame_task_errors')
    per_side=('hand_contact_active','hand_frames_w','hand_finger_joints','hand_link_contact_positions',
        'hand_link_contact_normals','hand_object_contact_positions','hand_object_contact_normals','hand_object_contact_part_ids')
    for key in temporal:
        if key not in row or row[key] is None:continue
        if len(row[key])!=frames:raise ValueError('Incompatible source clock: '+key)
        row[key]=row[key][start:stop]
    for key in per_side:
        if key not in row or row[key] is None:continue
        for side,values in enumerate(row[key]):
            if values is None or len(values)==0:continue
            if len(values)!=frames:raise ValueError('Incompatible per-side source clock: '+key)
            row[key][side]=values[start:stop]
    return row,dict(schema='real2sim2real.reference-window/1',source_frames=frames,
        start_native_frame=start,stop_native_frame_exclusive=stop,frames=stop-start,
        source_fps=original['fps'],source_start_s=start/original['fps'],source_end_s=(stop-1)/original['fps'],
        pose_values_unchanged=True,source_reference_modified=False,full_task_evaluation=False)


def prepare(reference,output,*,start,stop,hardlink=False):
    from pathlib import Path
    import os,json,shutil
    import pyarrow as pa,pyarrow.parquet as pq
    from real2sim2real.artifacts import load,publish,digest,write_json
    reference,output=Path(reference),Path(output)
    record=load(reference,kinds={'robot-reference'})
    if output.exists():raise FileExistsError('Preserve previous window')
    table=pq.read_table(reference/record['files']['reference']['path'])
    if table.num_rows!=1:raise ValueError('One motion_v1 sequence required')
    row,receipt=slice_row(table.to_pylist()[0],start,stop)
    # Copy only declared native assets. A packaged source can also contain
    # derived control-rate parquets, which must be regenerated for this window.
    output.mkdir(parents=True)
    for role,value in record['files'].items():
        if role=='reference':continue
        target=output/value['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if hardlink:os.link(reference/value['path'],target)
        else:shutil.copy2(reference/value['path'],target)
    target=output/record['files']['reference']['path'];target.parent.mkdir(parents=True,exist_ok=True)
    metadata=dict(table.schema.metadata or {});metadata[b'real2sim2real.reference_window']=json.dumps(receipt,sort_keys=True).encode()
    pq.write_table(pa.Table.from_pylist([row],schema=table.schema.with_metadata(metadata)),target)
    receipt['parent_artifact_sha256']=digest(reference/'artifact.json')
    write_json(output/'window.json',receipt)
    publish(output,'robot-reference',{**{key:output/value['path'] for key,value in record['files'].items()},'window':output/'window.json'},
        metadata=dict(record['metadata'],frames=stop-start,native_window=receipt,full_task_evaluation=False))
    return receipt
