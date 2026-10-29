"""Shared explicit runtime settings for standalone or pipeline object tracking."""
from pathlib import Path

MODEL_RUNS = ('2024-01-11-20-02-45', '2023-10-28-18-33-37')


def tracking_environment(weights):
    """Reject missing model files before allocating a GPU; never download weights.

    CLI-help/import checks alone cannot validate the model/cache location. This
    checks both score/refiner configurations and checkpoint presence; the caller
    may additionally deserialize trusted checkpoints on CPU before GPU launch.
    """
    root=Path(weights).resolve()
    for run in MODEL_RUNS:
        for name in ('config.yml','model_best.pth'):
            path=root/run/name
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError('Tracking requires explicit local model file: '+str(path))
    return dict(FOUNDATIONPOSE_WEIGHTS_DIR=str(root),
                OPENCV_IO_ENABLE_OPENEXR='1',PYOPENGL_PLATFORM='egl',
                R2S2R_FP_RENDER_BATCH='16',R2S2R_FP_REFINE_BATCH='32',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
