"""CPU-only bounded MANO articulation fitting; no image encoder or downloads.

The shape and metric scale are fixed. Only local joint rotations and a small
camera-space rigid correction may change. Held-out depth is never optimized.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def validate_options(options):
    defaults = dict(
        max_joint_correction_deg=30.0,
        max_rigid_correction_deg=15.0,
        max_translation_correction_m=0.03,
        correspondence_iterations=5,
    )
    if set(options) - set(defaults):
        raise ValueError("Unknown hand refinement optimizer option")
    result = dict(defaults, **options)
    for key, maximum in (
        ("max_joint_correction_deg", 45),
        ("max_rigid_correction_deg", 30),
        ("max_translation_correction_m", 0.05),
    ):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 < value <= maximum
        ):
            raise ValueError("Invalid bounded refinement option: " + key)
    count = result["correspondence_iterations"]
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("Require 1 to 20 correspondence iterations")
    return result


def bounded_vectors(parameters, maximum):
    import torch

    norm = torch.sqrt((parameters * parameters).sum(-1, keepdim=True) + 1e-12)
    return parameters * (maximum * torch.tanh(norm) / norm)


def rotation_matrices(vectors):
    import torch

    x, y, z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    return torch.matrix_exp(skew.reshape(*vectors.shape[:-1], 3, 3))


def damped_parameters(fit, initial_local, details, fraction):
    """Backtrack on rotation groups; never average hand vertices or joint poses."""
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Require a positive correction fraction no greater than one")
    local = np.asarray(details["incremental_local_rotation_delta_rad"]) * fraction
    rigid = np.asarray(details["rigid_rotation_delta_rad"]) * fraction
    offset = np.asarray(details["translation_delta_m"]) * fraction
    cumulative = (
        Rotation.from_rotvec(local) * Rotation.from_rotvec(initial_local)
    ).as_rotvec()
    updated = np.asarray(fit).copy()
    updated[:3] = (
        Rotation.from_rotvec(rigid) * Rotation.from_rotvec(fit[:3])
    ).as_rotvec()
    updated[3:6] += offset
    return updated, cumulative


class ManoHand:
    def __init__(self, model_path):
        import torch
        import smplx
        from smplx.vertex_ids import vertex_ids
        from .hamer_worker import legacy_mano_compatibility

        legacy_mano_compatibility()
        self.model = smplx.MANOLayer(
            str(Path(model_path)), is_rhand=True, use_pca=False
        ).eval()
        self.model.requires_grad_(False)
        self.tip_ids = torch.tensor(list(vertex_ids["mano"].values()), dtype=torch.long)
        self.joint_ids = [
            0,
            13,
            14,
            15,
            16,
            1,
            2,
            3,
            17,
            4,
            5,
            6,
            18,
            10,
            11,
            12,
            19,
            7,
            8,
            9,
            20,
        ]

    def __call__(self, betas, pose, orient, hand):
        import torch

        value = self.model(
            betas=betas[None],
            hand_pose=pose[None],
            global_orient=orient[None],
            pose2rot=False,
        )
        vertices = value.vertices[0]
        joints = torch.cat((value.joints[0], vertices[self.tip_ids]), 0)[self.joint_ids]
        reflection = vertices.new_tensor([2 * hand - 1, 1, 1])
        return (vertices - joints[:1]) * reflection, (joints - joints[:1]) * reflection


def refine(
    model,
    raw,
    fit,
    keypoints,
    points,
    k,
    hand,
    *,
    options=None,
    initial_local=None,
    joint_weights=None,
    temporal_target=None,
    temporal_weight=0.0,
):
    import torch
    import trimesh
    from .mesh_surface import MeshSurfaceQuery

    options = validate_options(options or {})
    torch.set_num_threads(2)
    tensor = lambda value: torch.as_tensor(
        np.asarray(value).copy(), dtype=torch.float32
    )
    beta, pose, orient = (tensor(raw[key]) for key in ("betas", "pose", "orient"))
    original_v, original_j = model(beta, pose, orient, hand)
    if not np.allclose(original_v.detach().numpy(), raw["vertices"], atol=2e-5, rtol=0):
        raise ValueError("MANO reconstruction differs from the retained hand prior")
    if not np.allclose(original_j.detach().numpy(), raw["joints"], atol=2e-5, rtol=0):
        raise ValueError("MANO joint order differs from the retained hand prior")
    if len(points) < 80 or not np.isfinite(keypoints).all():
        raise ValueError("Insufficient measured hand evidence")
    joint_bound = np.deg2rad(options["max_joint_correction_deg"])
    rigid_bound = np.deg2rad(options["max_rigid_correction_deg"])
    translation_bound = options["max_translation_correction_m"]
    camera_r = tensor(Rotation.from_rotvec(fit[:3]).as_matrix())
    camera_t = tensor(fit[3:6])
    scale = float(np.exp(fit[6]))
    intrinsic, target_uv = tensor(k), tensor(keypoints)
    has_initial_local, has_joint_weights = (
        initial_local is not None,
        joint_weights is not None,
    )
    initial_local = (
        np.zeros((15, 3)) if initial_local is None else np.asarray(initial_local)
    )
    joint_weights = np.ones(21) if joint_weights is None else np.asarray(joint_weights)
    if initial_local.shape != (15, 3) or not np.isfinite(initial_local).all():
        raise ValueError("Require finite local MANO corrections")
    if (
        joint_weights.shape != (21,)
        or not np.isfinite(joint_weights).all()
        or not ((joint_weights > 0) & (joint_weights <= 1)).all()
    ):
        raise ValueError("Require 21 positive image weights no greater than one")
    if not np.isfinite(temporal_weight) or not 0 <= temporal_weight <= 10:
        raise ValueError("Invalid temporal fitting weight")
    if temporal_target is not None:
        temporal_target = np.asarray(temporal_target)
        if temporal_target.shape != (21, 3) or not np.isfinite(temporal_target).all():
            raise ValueError("Temporal target must contain 21 finite camera joints")
        temporal_target = tensor(temporal_target)
    elif temporal_weight:
        raise ValueError("Temporal weight requires a measured neighbor target")
    base_local = rotation_matrices(tensor(initial_local))
    image_weights = tensor(joint_weights)[:, None]
    parameters = torch.zeros((17, 3), requires_grad=True)
    faces = np.asarray(raw["faces"])

    def geometry():
        local = bounded_vectors(parameters[:15], joint_bound)
        rigid = bounded_vectors(parameters[15], rigid_bound)
        offset = bounded_vectors(parameters[16], translation_bound)
        corrected_pose = (
            rotation_matrices(local) @ base_local @ pose
            if has_initial_local
            else rotation_matrices(local) @ pose
        )
        v, j = model(beta, corrected_pose, orient, hand)
        r = rotation_matrices(rigid) @ camera_r
        return (
            v * scale @ r.T + camera_t + offset,
            j * scale @ r.T + camera_t + offset,
            local,
            rigid,
            offset,
        )

    history = []
    for _ in range(options["correspondence_iterations"]):
        with torch.no_grad():
            vertices = geometry()[0].numpy()
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        closest, distances, face_ids = MeshSurfaceQuery(mesh).closest(points)
        use = distances <= min(0.08, max(0.01, np.quantile(distances, 0.8)))
        if use.sum() < 50:
            raise ValueError("Insufficient refinement surface correspondences")
        ids = torch.as_tensor(faces[face_ids[use]].copy(), dtype=torch.long)
        bary = tensor(
            trimesh.triangles.points_to_barycentric(
                mesh.triangles[face_ids[use]], closest[use]
            )
        )
        targets = tensor(points[use])
        optimizer = torch.optim.LBFGS(
            [parameters], lr=0.7, max_iter=20, line_search_fn="strong_wolfe"
        )

        def objective():
            optimizer.zero_grad()
            v, j, local, rigid, offset = geometry()
            projected = j @ intrinsic.T
            uv = projected[:, :2] / projected[:, 2:].clamp(min=0.05)
            surface = (v[ids] * bary[..., None]).sum(1)
            image_residual = torch.sqrt(1 + ((uv - target_uv) / 5).square()) - 1
            image_loss = (
                (image_residual * image_weights).sum() / (2 * image_weights.sum())
                if has_joint_weights
                else image_residual.mean()
            )
            depth_loss = (
                torch.sqrt(1 + ((surface - targets) / 0.006).square()) - 1
            ).mean()
            prior = (local / np.deg2rad(20)).square().mean()
            prior = (
                prior
                + (rigid / np.deg2rad(10)).square().mean()
                + (offset / 0.02).square().mean()
            )
            loss = image_loss + depth_loss + 0.1 * prior
            if temporal_target is not None:
                temporal_loss = (
                    torch.sqrt(1 + ((j - temporal_target) / 0.005).square()) - 1
                ).mean()
                loss = loss + temporal_weight * temporal_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite articulation objective")
            loss.backward()
            return loss

        optimizer.step(objective)
        history.append(float(objective().detach()))
    with torch.no_grad():
        v, j, local, rigid, offset = geometry()
    details = dict(
        objective=history,
        local_rotation_delta_rad=(
            Rotation.from_matrix(
                (rotation_matrices(local) @ base_local).numpy().astype(float)
            )
            .as_rotvec()
            .tolist()
            if has_initial_local
            else local.numpy().tolist()
        ),
        incremental_local_rotation_delta_rad=local.numpy().tolist(),
        rigid_rotation_delta_rad=rigid.numpy().tolist(),
        translation_delta_m=offset.numpy().tolist(),
        max_joint_correction_deg=float(
            np.rad2deg(np.linalg.norm(local.numpy(), axis=1)).max()
        ),
        rigid_correction_deg=float(np.rad2deg(np.linalg.norm(rigid.numpy()))),
        translation_correction_m=float(np.linalg.norm(offset.numpy())),
    )
    return v.numpy(), j.numpy(), details
