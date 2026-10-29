"""Pinned CPU TorchScript MHR adapter. Importing the base package needs no torch.

The decoder equations and COCO mapping follow the repository's CARI4D MHRLayer
and SAM 3D Body MHR head. Models/buffers are supplied under their original licenses.
"""

import json
from pathlib import Path

import numpy as np

from .bundle import file_sha256
from .human_contract import DecodedHuman, HumanParameters, REPAIR_CONTROLS

COCO17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


class TorchMHRDecoder:
    def __init__(self, spec_path: Path):
        import torch
        if torch.cuda.is_available():
            raise ValueError("Human repair is CPU-only; run with CUDA_VISIBLE_DEVICES='' and no GPU container access")
        torch.set_num_threads(2)
        spec_path = Path(spec_path)
        spec = json.loads(spec_path.read_text())
        if spec.get("schema") != "r2s2r_engines.native-mhr-decoder/1":
            raise ValueError("Unsupported decoder specification")
        paths = {}
        for name in ("model", "buffers"):
            path = Path(spec[name]["path"])
            path = path if path.is_absolute() else spec_path.parent / path
            if file_sha256(path) != spec[name]["sha256"]:
                raise ValueError(f"Decoder asset hash mismatch: {name}")
            paths[name] = path
        self.model = torch.jit.load(str(paths["model"]), map_location="cpu").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        buffers = torch.load(paths["buffers"], map_location="cpu", weights_only=True)
        self.parameter_names = list(self.model.get_parameter_names())
        if self.parameter_names[:6] != ["root_tx", "root_ty", "root_tz", "root_rx", "root_ry", "root_rz"]:
            raise ValueError("Unsupported native root layout")
        # The exported model lists 204 pose/scale controls followed by 45
        # identity controls; forward receives the latter as a separate tensor.
        if len(self.parameter_names) != 249 or any(name not in self.parameter_names[:68] for name in REPAIR_CONTROLS):
            raise ValueError("Unsupported native articulation layout")
        if self.parameter_names[130:136] != ["spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible",
                                             "arm_length_flexible", "hip_width_flexible", "leg_length_flexible"]:
            raise ValueError("Unsupported flexible-length layout")
        hands = sorted(buffers["hand_joint_idxs_left"].tolist() + buffers["hand_joint_idxs_right"].tolist())
        if hands != list(range(68, 122)):
            raise ValueError("Unexpected native finger controls")
        self.faces = buffers["faces"].numpy().astype(np.int64)
        self.mapping = buffers["keypoint_mapping"][COCO17].float()
        joint_names = list(self.model.get_joint_names())
        indices, weights = self.model.get_lbsw()
        dominant = indices.numpy()[np.arange(len(indices)), weights.numpy().argmax(1)]
        regions = []
        for prefix in ("l", "r"):
            joints = [i for i, name in enumerate(joint_names) if name == prefix + "_wrist" or
                      name.startswith(tuple(prefix + "_" + part for part in ("thumb", "index", "middle", "ring", "pinky")))]
            regions.append(np.flatnonzero(np.isin(dominant, joints)))
        if any(len(r) == 0 for r in regions):
            raise ValueError("Decoder lacks anatomical hand regions")
        self.hand_vertices = tuple(regions)
        self.identity = {"backend": "native-mhr-torchscript-cpu/1", "torch": torch.__version__,
                         "model_sha256": spec["model"]["sha256"], "buffers_sha256": spec["buffers"]["sha256"],
                         "root_translation_m_per_unit": 0.1, "decoded_length_m_per_unit": 0.01,
                         "parameter_names": self.parameter_names}

    def decode_tensor(self, pose, shape, scales, expression, native_to_world):
        """Differentiable equations; called only inside the optional CPU backend."""
        import torch
        n = len(pose)
        def expanded(value):
            return torch.as_tensor(np.array(value), dtype=torch.float32).expand(n, -1)
        vertices, joints = self.model(expanded(shape), torch.cat([pose, expanded(scales)], dim=1), expanded(expression))
        points = torch.cat([vertices, joints[..., :3]], dim=1) * .01
        keypoints = torch.einsum("jv,tvc->tjc", self.mapping, points)
        transform = torch.as_tensor(native_to_world, dtype=pose.dtype)
        return vertices * .01 @ transform[:3, :3].T + transform[:3, 3], keypoints @ transform[:3, :3].T + transform[:3, 3]

    def decode(self, parameters: HumanParameters, native_to_world: np.ndarray) -> DecodedHuman:
        import torch
        vertices, joints = [], []
        with torch.inference_mode():
            for start in range(0, len(parameters.pose), 16):
                stop = min(start + 16, len(parameters.pose))
                def part(a):
                    return a if a.ndim == 1 else a[start:stop]
                v, j = self.decode_tensor(torch.tensor(parameters.pose[start:stop], dtype=torch.float32),
                                          part(parameters.shape), part(parameters.scales), part(parameters.expression), native_to_world)
                vertices.append(v.numpy()); joints.append(j.numpy())
        result = DecodedHuman(np.concatenate(vertices), np.concatenate(joints))
        if not np.isfinite(result.vertices_world).all() or not np.isfinite(result.joints_world).all():
            raise ValueError("Native decoder produced nonfinite geometry")
        return result
