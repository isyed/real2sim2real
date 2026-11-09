import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from real2sim2real.artifacts import digest,write_json
from real2sim2real.reconstruction.mask_review import audit_mask_timelines,object_visibility_warnings
from real2sim2real.reconstruction.mask_acceptance import accept,enforce,MaskReviewRequired


def fixture(tmp_path,counts=(8,8)):
    masks=tmp_path/'masks'
    for role in (0,1):
        folder=masks/str(role);folder.mkdir(parents=True)
        for i,count in enumerate(counts):
            pixels=np.zeros((6,8),np.uint8);pixels.flat[:30 if role==0 else count]=255
            Image.fromarray(pixels).save(folder/f'{i:06d}.png')
    prompts=tmp_path/'prompts.json';prompts.write_text('{"prompts":[]}')
    video=tmp_path/'video.mp4';video.write_bytes(b'RGB fixture');video_hash=digest(video)
    audit=audit_mask_timelines(masks,len(counts));review=tmp_path/'mask_review/mask_review.json'
    write_json(review,dict(schema='real2sim2real.mask-review/1',source_frames=len(counts),source_video_sha256=video_hash,
        prompts_sha256=digest(prompts),mask_timelines_complete=True,reviewed_frame_indices=[0],all_frame_audit=audit))
    receipt=review.with_name('acceptance.json')
    request=dict(input=str(video),output=str(tmp_path),config_base=str(tmp_path),input_sha256=video_hash,options={'prompts':str(prompts)})
    return masks,prompts,review,receipt,request


def approve(f,**kwargs):
    masks,prompts,review,receipt,request=f
    return accept(review,masks,receipt,reviewer='operator',human_accepted=True,object_accepted=True,**kwargs)


def test_default_gate_requires_actual_subject_acceptance(tmp_path):
    f=fixture(tmp_path)
    with pytest.raises(MaskReviewRequired,match='explicit mask acceptance'):enforce(f[4],'metric_mesh')
    approve(f)
    assert enforce(f[4],'metric_mesh')['object_accepted'] is True
    with pytest.raises(FileExistsError):approve(f)


def test_unsampled_mask_change_invalidates_acceptance(tmp_path):
    f=fixture(tmp_path);approve(f)
    Image.fromarray(np.full((6,8),255,np.uint8)).save(f[0]/'1/000001.png')
    with pytest.raises(MaskReviewRequired,match='Masks changed'):enforce(f[4],'object_motion')


def test_video_and_prompt_changes_are_not_silently_accepted(tmp_path):
    f=fixture(tmp_path);approve(f)
    bad=dict(f[4],input_sha256='b'*64)
    with pytest.raises(MaskReviewRequired,match='video changed'):enforce(bad,'export_human')
    f[1].write_text('{"prompts":["changed"]}')
    with pytest.raises(MaskReviewRequired,match='Prompts changed'):enforce(f[4],'metric_mesh')


def test_missing_and_shrinking_masks_are_hints_not_semantic_truth(tmp_path):
    warning=object_visibility_warnings([40,8,0])
    assert warning['warning_codes']==['object_mask_missing','object_mask_shrinks']
    assert not warning['semantic_identity_verified']
    f=fixture(tmp_path,counts=(30,5,0))
    with pytest.raises(MaskReviewRequired,match='acknowledge'):approve(f)
    approve(f,acknowledge_warnings=['object_mask_missing','object_mask_shrinks'])
    assert enforce(f[4],'metric_mesh')['object_accepted']


def test_first_frame_empty_cannot_generate_a_metric_mesh(tmp_path):
    f=fixture(tmp_path,counts=(0,8))
    with pytest.raises(MaskReviewRequired,match='Frame-zero'):approve(f,acknowledge_warnings=['object_mask_missing'])


def test_unreviewed_mode_is_explicit_and_recorded(tmp_path):
    f=fixture(tmp_path);f[4]['options']['mask_review_policy']='unreviewed'
    proof=enforce(f[4],'metric_mesh')
    assert proof['subject_identity_accepted'] is False
    assert (tmp_path/'mask_review/unreviewed_run.json').is_file()
    f[4]['options']['mask_review_policy']='typo'
    with pytest.raises(ValueError,match='Unknown'):enforce(f[4],'metric_mesh')


def test_dependency_checks_and_segmentation_need_no_prior_acceptance(tmp_path):
    f=fixture(tmp_path)
    assert enforce(f[4],'segment') is None
    assert enforce(f[4],'metric_mesh',check_backend=True) is None


def test_subjects_cannot_be_accepted_by_truthy_strings(tmp_path):
    f=fixture(tmp_path)
    with pytest.raises(MaskReviewRequired):accept(f[2],f[0],f[3],reviewer='operator',human_accepted='yes',object_accepted=True)


def test_malformed_numeric_acceptance_is_rejected(tmp_path):
    f=fixture(tmp_path);approve(f)
    record=json.loads(f[3].read_text());record['object_accepted']=1;write_json(f[3],record)
    with pytest.raises(MaskReviewRequired):enforce(f[4],'metric_mesh')


def test_changed_rgb_file_is_rejected_even_if_request_hash_was_not_updated(tmp_path):
    f=fixture(tmp_path);approve(f)
    Path(f[4]['input']).write_bytes(b'changed RGB')
    with pytest.raises(MaskReviewRequired,match='Source video changed'):enforce(f[4],'object_motion')
