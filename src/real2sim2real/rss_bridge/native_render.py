"""CPU RSS rendering for source-pose inspection, never a physical rollout.

Poses are written to renderer state deliberately: this is reconstruction
measurement, not simulated object motion or evidence of a robot grasp.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from real2sim2real.artifacts import load
from .bundle import rigid
from .scene import inspection_scene


class SourceRenderer:
    def __init__(self, bundle, *, frame_id, roi, cache_dir):
        import warp as wp
        import robosimstudio as rss
        from robosimstudio.specs.cell import CellSpec
        from robosimstudio.scene.registry import get_cell

        wp.config.kernel_cache_dir = str(Path(cache_dir).resolve())
        self.bundle = Path(bundle)
        self.receipt = load(bundle, kinds={"rss-source-exchange"})
        self.transform = np.asarray(self.receipt["metadata"]["source_to_rss"])
        self.scene, original_camera = inspection_scene(bundle, frame_id)
        # RSS's void cell has an automatic z=0 ground plane, which occludes
        # camera-space inputs below zero. For inspection, explicitly move only
        # that synthetic plane away. This is not a calibrated task support.
        name = "r2s2r_source_inspection_ground_minus100_v1"
        try:
            cell = get_cell(name)
        except KeyError:
            cell = rss.register_cell(CellSpec(name=name, ground_z=-100., surface_z=0.,
                provenance="Synthetic inspection floor; no physical support inference"))
        if cell.ground_z != -100. or cell.surface_z != 0. or cell.statics or cell.asset:
            raise ValueError("Inspection cell registration has incompatible geometry")
        self.scene.cell = name
        x0, y0, x1, y1 = map(int, roi)
        if not (0 <= x0 < x1 <= original_camera.width and 0 <= y0 < y1 <= original_camera.height):
            raise ValueError("ROI must be inside the source image")
        self.camera = replace(original_camera, width=x1-x0, height=y1-y0,
                              cx=original_camera.cx-x0, cy=original_camera.cy-y0)
        self.sim = self.scene.build(device="cpu", use_cuda_graph=False,
                                    renderer="tiled", warmup=False)
        bodies = self.sim["source_object"].bodies
        if len(bodies) != 1:
            self.close()
            raise ValueError("Inspection requires exactly one object body")
        self.body = int(bodies[0])
        labels = self.sim.scene.handles.segmentation_labels()
        shapes = self.sim["source_object"].shapes
        self.labels = {int(labels[int(shape)]) for shape in shapes}

    def render(self, pose_source):
        pose = self.transform @ rigid(pose_source)
        if pose[2, 3] < -90:
            raise ValueError("Source is too close to the synthetic inspection ground")
        state = self.sim.state.body_q.numpy()
        state[self.body] = np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()]
        self.sim.state.body_q.assign(state)
        frame = self.sim.render(self.camera, channels=("rgb", "seg", "depth"))
        frame["object_mask"] = np.isin(frame["seg"], list(self.labels))
        if not frame["object_mask"].any():
            raise ValueError("RSS rendered no object pixels; check coordinates and occlusion")
        return frame

    def close(self):
        if getattr(self, "sim", None) is not None:
            self.sim.close()
            self.sim = None
