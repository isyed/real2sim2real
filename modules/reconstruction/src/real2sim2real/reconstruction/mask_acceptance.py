"""Explicit subject review bound to the video, prompts and all propagated masks.

This records an operator decision. It cannot infer semantic identity or complete
object coverage from a bounding box, pixel count, or the model's own IoU.
"""
from pathlib import Path
import json
from real2sim2real.artifacts import digest,write_json


class MaskReviewRequired(ValueError):
    """Stop before dependent inference; retain the review for operator inspection."""


def _current(review_file,masks_root,*,source_video_sha256=None,prompts_file=None):
    from .mask_review import mask_set_digest
    review_file=Path(review_file)
    if not review_file.is_file():raise MaskReviewRequired('Mask review is missing; run pack_masks and inspect the contact sheet')
    review=json.loads(review_file.read_text())
    if review.get('schema')!='real2sim2real.mask-review/1' or review.get('mask_timelines_complete') is not True:raise MaskReviewRequired('Complete mask review required')
    audit=review.get('all_frame_audit',{})
    if audit.get('model_prompt_inputs_valid') is not True:raise MaskReviewRequired('Human mask inputs fail the existing model minimum')
    current=mask_set_digest(masks_root,review['source_frames'])
    if not audit.get('mask_set_sha256') or current!=audit['mask_set_sha256']:raise MaskReviewRequired('Masks changed or old review lacks complete hashes; regenerate mask review')
    if source_video_sha256 is not None and review.get('source_video_sha256')!=source_video_sha256:raise MaskReviewRequired('Reviewed video differs from reconstruction source')
    if prompts_file is not None and review.get('prompts_sha256')!=digest(prompts_file):raise MaskReviewRequired('Prompts changed after mask review')
    counts=audit.get('foreground_counts',{}).get('object',[])
    if not counts or counts[0]<=0:raise MaskReviewRequired('Frame-zero object mask is empty; this mesh recipe needs a visible first-frame object')
    return review


def accept(review_file,masks_root,output,*,reviewer,human_accepted,object_accepted,acknowledge_warnings=()):
    review=_current(review_file,masks_root)
    if not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('Name the reviewer')
    if human_accepted is not True or object_accepted is not True:raise MaskReviewRequired('Both intended subjects require explicit acceptance')
    warnings=set(review['all_frame_audit'].get('object_visibility',{}).get('warning_codes',[]))
    if not warnings.issubset(set(acknowledge_warnings)):raise MaskReviewRequired('Inspect and acknowledge mask visibility/shrink warnings: '+', '.join(sorted(warnings)))
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve the previous acceptance; use a fresh receipt')
    record=dict(schema='real2sim2real.mask-acceptance/1',review_sha256=digest(review_file),
        source_video_sha256=review['source_video_sha256'],prompts_sha256=review['prompts_sha256'],
        mask_set_sha256=review['all_frame_audit']['mask_set_sha256'],reviewed_frame_indices=review['reviewed_frame_indices'],
        reviewer=reviewer.strip(),human_accepted=True,object_accepted=True,acknowledged_warnings=sorted(set(acknowledge_warnings)),
        physical_validity_established=False,acceptance_scope='Operator review of visible subject identity and coverage; not ground truth')
    write_json(output,record);return record


def verify(review_file,acceptance_file,masks_root,*,source_video_sha256,prompts_file):
    review=_current(review_file,masks_root,source_video_sha256=source_video_sha256,prompts_file=prompts_file)
    acceptance_file=Path(acceptance_file)
    if not acceptance_file.is_file():raise MaskReviewRequired('Inspect mask_review/contact_sheet.jpg, then write an explicit mask acceptance receipt before continuing')
    approved=json.loads(acceptance_file.read_text())
    expected=dict(schema='real2sim2real.mask-acceptance/1',review_sha256=digest(review_file),source_video_sha256=review['source_video_sha256'],prompts_sha256=review['prompts_sha256'],mask_set_sha256=review['all_frame_audit']['mask_set_sha256'],reviewed_frame_indices=review['reviewed_frame_indices'],human_accepted=True,object_accepted=True)
    if any(approved.get(k)!=v for k,v in expected.items()) or approved.get('human_accepted') is not True or approved.get('object_accepted') is not True or not isinstance(approved.get('reviewer'),str) or not approved['reviewer'].strip():raise MaskReviewRequired('Acceptance does not match this video, review, subjects and mask set')
    if not set(review['all_frame_audit'].get('object_visibility',{}).get('warning_codes',[])).issubset(set(approved.get('acknowledged_warnings',[]))):raise MaskReviewRequired('New visibility warnings need explicit review')
    return approved


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',required=True,type=Path);p.add_argument('--masks',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reviewer',required=True);p.add_argument('--human-accepted',action='store_true');p.add_argument('--object-accepted',action='store_true')
    p.add_argument('--acknowledge',nargs='*',default=[]);a=p.parse_args()
    print(json.dumps(accept(a.review,a.masks,a.output,reviewer=a.reviewer,human_accepted=a.human_accepted,object_accepted=a.object_accepted,acknowledge_warnings=a.acknowledge),indent=2))


if __name__=='__main__':main()


def enforce(request,phase,*,check_backend=False):
    """Default review gate for stages that consume the selected subjects."""
    from real2sim2real.stage import option_path
    policy=request['options'].get('mask_review_policy','require_acceptance')
    if policy not in ('require_acceptance','unreviewed'):raise ValueError('Unknown mask_review_policy')
    if check_backend or phase not in ('export_human','metric_mesh','object_motion'):return None
    output=Path(request['output']);prompts=option_path(request,'prompts')
    if digest(request['input'])!=request['input_sha256']:raise MaskReviewRequired('Source video changed after request creation')
    if policy=='unreviewed':
        record=dict(policy=policy,source_video_sha256=request['input_sha256'],prompts_sha256=digest(prompts),
            subject_identity_accepted=False,warning='Explicit unreviewed run; semantic subject identity and full visible object coverage have not been accepted')
        write_json(output/'mask_review/unreviewed_run.json',record);return record
    approved=request['options'].get('mask_acceptance')
    path=Path(approved) if approved is not None else output/'mask_review/acceptance.json'
    if not path.is_absolute():path=Path(request['config_base'])/path
    return verify(output/'mask_review/mask_review.json',path,output/'masks',source_video_sha256=request['input_sha256'],prompts_file=prompts)
