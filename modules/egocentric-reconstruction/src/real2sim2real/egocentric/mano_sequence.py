"""CPU sequence-level MANO fitting with fixed metric shape, scene and anchors."""

import numpy as np
from scipy.spatial.transform import Rotation

from .hand_sequence import affected_edges, sequence_options, protected_boundary_edges
from .mano_refinement import bounded_vectors, rotation_matrices, image_evidence


def fit_sequence(
    model,
    prior,
    fits,
    keypoints,
    weights,
    points,
    k,
    hand,
    world_from_camera,
    initial_world_joints,
    selected,
    times,
    *,
    options,
    baseline_mask=None,
):
    import torch
    import trimesh
    from .mesh_surface import MeshSurfaceQuery

    options = sequence_options(options)
    torch.set_num_threads(2)
    tensor = lambda x: torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32)
    selected = np.asarray(selected, bool)
    ids = np.flatnonzero(selected)
    if not len(ids) or len(prior["pose"]) != len(ids):
        raise ValueError("MANO priors must match selected sequence frames")
    dt = np.diff(times)
    if not (dt > 0).all() or (dt > options["max_step_s"]).any():
        raise ValueError("Sequence optimization cannot bridge capture gaps")
    edges, triples = affected_edges(selected)
    if not triples.any():
        raise ValueError("Sequence optimization needs observed temporal triples")
    if any(len(p) < 80 or not np.isfinite(p).all() for p in points):
        raise ValueError("Every variable frame requires measured fitting depth")
    evidence = [image_evidence(uv, w) for uv, w in zip(keypoints, weights)]
    target_uv = tensor(np.stack([x[0] for x in evidence]))
    image_weights = tensor(np.stack([x[1] for x in evidence]))
    baseline_mask = (
        np.zeros(len(selected), bool)
        if baseline_mask is None
        else np.asarray(baseline_mask, bool)
    )
    protected = protected_boundary_edges(selected, baseline_mask)
    data_weights = tensor(
        np.where(baseline_mask[ids], options["anchor_data_weight"], 1)
    )
    beta, pose, orient = [tensor(prior[name]) for name in ("betas", "pose", "orient")]
    camera_r = tensor(Rotation.from_rotvec(fits[:, :3]).as_matrix())
    camera_t, scale = tensor(fits[:, 3:6]), tensor(np.exp(fits[:, 6]))[:, None, None]
    wr, wt = (
        tensor(world_from_camera[ids, :3, :3]),
        tensor(world_from_camera[ids, :3, 3]),
    )
    base_world = tensor(initial_world_joints)
    index = torch.tensor(ids, dtype=torch.long)
    time_step = tensor(dt)[:, None, None]
    parent_velocity = torch.diff(base_world, dim=0) / time_step
    intrinsic = tensor(k)
    params = torch.zeros((len(ids), 17, 3), requires_grad=True)
    bounds = (
        np.deg2rad(options["max_joint_delta_deg"]),
        np.deg2rad(options["max_rigid_delta_deg"]),
        options["max_translation_delta_m"],
    )

    def geometry():
        local = bounded_vectors(params[:, :15], bounds[0])
        rigid = bounded_vectors(params[:, 15], bounds[1])
        offset = bounded_vectors(params[:, 16], bounds[2])
        v, j = model.batch(beta, rotation_matrices(local) @ pose, orient, hand)
        r = rotation_matrices(rigid) @ camera_r
        v = v * scale @ r.transpose(-1, -2) + camera_t[:, None] + offset[:, None]
        j = j * scale @ r.transpose(-1, -2) + camera_t[:, None] + offset[:, None]
        world = j @ wr.transpose(-1, -2) + wt[:, None]
        return v, j, base_world.index_copy(0, index, world), local, rigid, offset

    with torch.no_grad():
        v, j, world, *_ = geometry()
        if not np.allclose(
            v.numpy(), prior["camera_vertices"], atol=2e-5, rtol=0
        ) or not np.allclose(j.numpy(), prior["camera_joints"], atol=2e-5, rtol=0):
            raise ValueError(
                "Sequence MANO parameters do not reproduce retained geometry"
            )
        if not np.allclose(world.numpy(), initial_world_joints, atol=2e-5, rtol=0):
            raise ValueError("Sequence world-frame initialization differs")
    history = []
    faces = np.asarray(prior["faces"])
    for outer in range(options["correspondence_iterations"]):
        with torch.no_grad():
            current = geometry()[0].numpy()
        correspondences = []
        for i, (vertices, measured) in enumerate(zip(current, points)):
            mesh = trimesh.Trimesh(vertices, faces, process=False)
            closest, distances, face_ids = MeshSurfaceQuery(mesh).closest(measured)
            use = distances <= min(0.08, max(0.01, np.quantile(distances, 0.8)))
            if use.sum() < 50:
                raise ValueError("Insufficient sequence surface correspondences")
            tri_ids = torch.as_tensor(faces[face_ids[use]].copy(), dtype=torch.long)
            bary = tensor(
                trimesh.triangles.points_to_barycentric(
                    mesh.triangles[face_ids[use]], closest[use]
                )
            )
            correspondences.append((tri_ids, bary, tensor(measured[use])))
        optimizer = torch.optim.LBFGS(
            [params], lr=0.7, max_iter=30, line_search_fn="strong_wolfe"
        )
        metrics = {}

        def objective():
            optimizer.zero_grad()
            v, j, world, local, rigid, offset = geometry()
            projection = j @ intrinsic.T
            uv = projection[..., :2] / projection[..., 2:].clamp(min=0.05)
            robust = lambda x: torch.sqrt(1 + x.square()) - 1
            image_by_frame = (
                robust((uv - target_uv) / 5) * image_weights[..., None]
            ).sum((1, 2)) / (2 * image_weights.sum(1))
            image_loss = (image_by_frame * data_weights).mean()
            surface_losses = []
            for i, (tri, bary, measured) in enumerate(correspondences):
                surface = (v[i, tri] * bary[..., None]).sum(1)
                surface_losses.append(robust((surface - measured) / 0.006).mean())
            depth_loss = (torch.stack(surface_losses) * data_weights).mean()
            velocity = torch.diff(world, dim=0) / time_step
            acceleration = (
                2 * torch.diff(velocity, dim=0) / (time_step[1:] + time_step[:-1])
            )
            temporal = robust(
                acceleration[triples] / options["acceleration_scale_m_s2"]
            ).mean()
            temporal = (
                temporal
                + 0.1 * robust(velocity[edges] / options["velocity_scale_m_s"]).mean()
            )
            boundary_loss = (
                robust((velocity[protected] - parent_velocity[protected]) / 0.2).mean()
                if protected.any()
                else params.sum() * 0
            )
            prior_loss = (local / np.deg2rad(20)).square().mean()
            prior_loss += (rigid / np.deg2rad(10)).square().mean() + (
                offset / 0.02
            ).square().mean()
            loss = (
                image_loss
                + depth_loss
                + 0.1 * prior_loss
                + options["temporal_weight"] * temporal
                + options["boundary_velocity_weight"] * boundary_loss
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite sequence objective")
            loss.backward()
            metrics.update(
                image=float(image_loss.detach()),
                depth=float(depth_loss.detach()),
                temporal=float(temporal.detach()),
                boundary=float(boundary_loss.detach()),
                total=float(loss.detach()),
            )
            return loss

        optimizer.step(objective)
        objective()
        history.append(dict(metrics))
        print(f"Sequence hand {hand}, iteration {outer + 1}: {metrics}", flush=True)
    with torch.no_grad():
        vertices, joints, world, local, rigid, offset = [x.numpy() for x in geometry()]
    new_fits = fits.copy()
    new_fits[:, :3] = (
        Rotation.from_rotvec(rigid) * Rotation.from_rotvec(fits[:, :3])
    ).as_rotvec()
    new_fits[:, 3:6] += offset
    return dict(
        camera_vertices=vertices,
        camera_joints=joints,
        world_joints=world,
        fits=new_fits,
        pose=Rotation.from_rotvec(local.reshape(-1, 3))
        .as_matrix()
        .reshape(-1, 15, 3, 3)
        @ prior["pose"],
        local_delta_rad=local,
        rigid_delta_rad=rigid,
        translation_delta_m=offset,
        objective=history,
    )
