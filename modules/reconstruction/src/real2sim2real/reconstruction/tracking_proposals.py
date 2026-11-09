"""Generate unaccepted pose proposals for subsequent multi-frame validation.

Requires a hash-bound GPU-accepted registration. Later tracker outputs are not
independently accepted states. Never use this output directly as a reference.
"""
import argparse
import json
from pathlib import Path

from real2sim2real.artifacts import digest,write_json


def verify_seed(seed, report):
    seed,report=map(Path,(seed,report))
    s=json.loads(seed.read_text());r=json.loads(report.read_text())
    if (s.get('source')!='observed_rgb_silhouette_depth_fit' or s.get('frame')!=0
            or s.get('clean_trajectory_used') is not False
            or s.get('probe_report_sha256')!=digest(report) or r.get('accepted') is not True
            or r.get('selected_index') is None or r.get('frame')!=0
            or r.get('schema')!='real2sim2real.initial-registration/1'
            or r.get('thresholds_relaxed') is not False):
        raise ValueError('Require hash-bound accepted registration evidence')
    return s,r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export','depth','seed','registration-report','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();verify_seed(a.seed,a.registration_report)
    if a.output.exists() or (a.export/'poses.npy').exists():
        raise ValueError('Fresh output and prediction-only inputs required')
    from .tracking_recovery import load_exporter,install_rgb_mode
    up=load_exporter();install_rgb_mode(up)
    original=up._atomic_pickle_dump
    def dump(path,value):
        if 'fp_poses_cam' in value:
            value['metadata'].update(source='unaccepted_rgb_tracking_proposals',
                proposal_only=True,independent_tracking_validation=False,
                tracking_refiner_mode='rgb_only',full_reference_accepted=False,
                proposal_adapter_sha256=digest(__file__),registration_report_sha256=digest(a.registration_report))
        original(path,value)
    up._atomic_pickle_dump=dump
    try:
        up.run_foundationpose_mhr_export(a.export,a.depth,a.output,camera_id=0,start=0,
            iteration=5,max_attempts=1,first_usable_frame_gt_rotation_oracle=False,
            sequence_mode=up.FOUNDATIONPOSE_SEQUENCE_MODE_REGISTER_FIRST_THEN_TRACK,
            observed_initial_pose=a.seed,rgb_only_score=True,viz_file=None)
        write_json(a.output.with_suffix('.proposal_receipt.json'),dict(proposal_only=True,
            output_sha256=digest(a.output),registration_report_sha256=digest(a.registration_report),
            seed_sha256=digest(a.seed),full_reference_accepted=False,
            requires_multi_frame_image_depth_support_and_motion_validation=True))
    finally:up._atomic_pickle_dump=original


if __name__=='__main__':main()
