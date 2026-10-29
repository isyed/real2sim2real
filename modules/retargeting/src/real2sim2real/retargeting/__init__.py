"""Configurable whole-body retargeting with explicit external assets."""
import os
from pathlib import Path
ASSETS_DIR=Path(os.environ.get("R2S2R_ASSETS", str(Path.home()/".cache/real2sim2real/assets")))
BODY_MODELS_DIR=ASSETS_DIR/"body_models"
G1_URDF_DIR=ASSETS_DIR/"urdfs/g1"
HUMAN_MOTION_DATA_DIR=Path(os.environ.get("R2S2R_MOTION_ROOT", "runs/motion"))
