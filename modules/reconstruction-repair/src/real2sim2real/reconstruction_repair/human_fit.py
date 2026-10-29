"""Optional bounded native-MHR fitting inside the pinned CPU torch runtime."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from .bundle import file_sha256, json_atomic
from .human_proposals import human_bounded_pose, human_controls, human_segments


def human_fit_proposal(bundle, parameters, decoded, decoder, transform, config, folder: Path, identity: str):
    import torch
    import torch.nn.functional as functional
    if not hasattr(decoder, "decode_tensor"):
        raise ValueError("Constrained fitting requires a differentiable native-MHR decoder")
    o, g = bundle.observations, bundle.geometry
    if "human_fit_frames" not in o:
        raise ValueError("Human fitting requires an explicit fit/holdout partition")
    fit = np.flatnonzero(o["human_fit_frames"].astype(bool))
    if not 2 <= len(fit) <= 64:
        raise ValueError("Bounded human fit supports 2–64 declared fitting frames")
    controls = human_controls(decoder)
    allowed = np.r_[0:3, controls]
    n = len(parameters.pose)
    columns = []
    for segment in human_segments(bundle, config):
        if len(segment) < 3:
            continue
        knots = [int(segment[0])]
        for i in segment[1:-1]:
            if g["timestamps_s"][i]-g["timestamps_s"][knots[-1]] >= config.human_fit_knot_spacing_s:
                knots.append(int(i))
        knots.append(int(segment[-1]))
        # A short segment still needs one interior control.
        if len(knots) == 2:
            knots.insert(1, int(segment[len(segment)//2]))
        for k in range(1, len(knots)-1):
            weights = np.zeros(n)
            a, center, b = knots[k-1:k+2]
            t = g["timestamps_s"]
            weights[a:center+1] = (t[a:center+1]-t[a])/(t[center]-t[a])
            weights[center:b+1] = (t[b]-t[center:b+1])/(t[b]-t[center])
            columns.append(weights*weights*(3-2*weights))
    if not columns:
        return parameters.pose.copy(), {"updates": 0, "reason": "No editable human segment"}
    basis = torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)
    bounds = torch.tensor([config.human_root_bound_m/.1]*3 + [np.deg2rad(config.human_angle_bound_deg)]*len(controls), dtype=torch.float32)
    raw = torch.zeros((basis.shape[1], len(allowed)), requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=config.human_fit_learning_rate)
    original = torch.tensor(parameters.pose, dtype=torch.float32)
    selector = torch.zeros((len(allowed), 136)); selector[torch.arange(len(allowed)), allowed] = 1
    w, k = torch.tensor(g["world_to_camera"], dtype=torch.float32), torch.tensor(g["intrinsics"], dtype=torch.float32)
    targets = torch.tensor(o["human_keypoints_xy"], dtype=torch.float32)
    confidence = torch.tensor(o["human_keypoints_confidence"], dtype=torch.float32)
    confidence[:, :5] = 0
    confidence *= confidence >= config.human_confidence_min
    mask = o["human_mask"].astype(bool)
    occ = o.get("human_occluder_mask", np.zeros_like(mask)).astype(bool)
    # Occluded pixels are neutral; full visible silhouettes are independently
    # rasterized in acceptance, not replaced by this sampled fitting surrogate.
    distance = np.stack([distance_transform_edt(~(mask[i] | occ[i])) for i in fit])
    maps = torch.tensor(distance[:, None], dtype=torch.float32)
    vertex_ids = np.linspace(0, decoded.vertices_world.shape[1]-1, min(512, decoded.vertices_world.shape[1])).astype(int)
    history, begin = [], 0
    receipt_path = folder / "fit_state.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        filename = f"fit_state_{int(receipt['update']):06d}.pt"
        state_path = folder / filename
        if receipt["identity"] != identity or file_sha256(state_path) != receipt["sha256"]:
            raise ValueError("Human optimizer checkpoint identity or checksum differs")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        with torch.no_grad(): raw.copy_(state["raw"])
        optimizer.load_state_dict(state["optimizer"]); begin = state["update"]; history = state["history"]
    for update in range(begin, config.human_fit_updates):
        optimizer.zero_grad()
        total = 0.0
        # Limit each autograd graph to eight human meshes. Priors and image
        # losses use fitting frames only; independent review covers all frames.
        for chunk_start in range(0, len(fit), 8):
            ids = fit[chunk_start:chunk_start+8]
            delta = (basis[ids] @ torch.tanh(raw))*bounds
            pose = original[ids] + delta @ selector
            def part(value): return value if value.ndim == 1 else value[ids]
            vertices, joints = decoder.decode_tensor(pose, part(parameters.shape), part(parameters.scales), part(parameters.expression), transform)
            camera_joints = torch.einsum("tij,tkj->tki", w[ids, :3, :3], joints)+w[ids, None, :3, 3]
            q = torch.einsum("tij,tkj->tki", k[ids], camera_joints)
            xy = q[..., :2]/q[..., 2:].clamp_min(1e-5)
            residual = functional.smooth_l1_loss((xy-targets[ids])/5, torch.zeros_like(xy), reduction="none").sum(-1)
            loss = (residual*confidence[ids]).sum()/confidence[ids].sum().clamp_min(1)
            camera_vertices = torch.einsum("tij,tkj->tki", w[ids, :3, :3], vertices[:, vertex_ids])+w[ids, None, :3, 3]
            qv = torch.einsum("tij,tkj->tki", k[ids], camera_vertices)
            h, width = mask.shape[1:]
            projected = qv[..., :2]/qv[..., 2:].clamp_min(1e-5)
            grid = 2*(projected+.5)/torch.tensor([width, h])-1
            outside = functional.grid_sample(maps[chunk_start:chunk_start+len(ids)], grid[:, None], align_corners=False, padding_mode="border")
            loss = loss + .2*(outside/5).square().mean()
            base_joints = torch.tensor(decoded.joints_world[ids], dtype=torch.float32)
            loss = loss + .05*((joints-base_joints)/config.human_joint_bound_m).square().mean()
            for side, indices in enumerate(decoder.hand_vertices):
                base_hand = torch.tensor(decoded.vertices_world[ids][:, indices], dtype=torch.float32)
                slack = np.where(o["human_contact_active"][ids, side], min(config.human_hand_bound_m, config.max_contact_gap_increase_m), config.human_hand_bound_m)
                movement = torch.linalg.vector_norm(vertices[:, indices]-base_hand, dim=-1)
                loss = loss + .2*(movement/torch.tensor(slack[:, None], dtype=torch.float32)).square().mean()
            loss = loss * (len(ids)/len(fit))
            if not torch.isfinite(loss): raise ValueError("Nonfinite native-MHR fitting objective")
            loss.backward(); total += float(loss.detach())
        prior = .01*torch.tanh(raw).square().mean()
        prior.backward(); optimizer.step()
        if (update+1)%config.human_fit_checkpoint_every == 0 or update+1 == config.human_fit_updates:
            history.append({"update": update+1, "loss_before_update": total})
            # A generation file plus atomic receipt retains the last committed
            # state if interrupted between writing the state and its receipt.
            state_path = folder / f"fit_state_{update+1:06d}.pt"
            temporary = state_path.with_suffix(".tmp")
            torch.save({"raw": raw.detach(), "optimizer": optimizer.state_dict(), "update": update+1, "history": history}, temporary)
            temporary.replace(state_path)
            json_atomic(receipt_path, {"identity": identity, "sha256": file_sha256(state_path), "update": update+1})
            json_atomic(folder/"status.json", {"status": "fitting", "update": update+1, "total_updates": config.human_fit_updates})
    with torch.no_grad(): delta = ((basis @ torch.tanh(raw))*bounds).numpy()
    candidate = parameters.pose.copy()
    candidate[:, allowed] += delta
    return human_bounded_pose(parameters.pose, candidate, controls, config), {
        "updates": config.human_fit_updates, "fit_frame_ids": g["frame_ids"][fit].tolist(),
        "holdout_frame_ids": g["frame_ids"][~o["human_fit_frames"].astype(bool)].tolist(), "history": history,
        "objective": "COCO reprojection, sampled visible-mask exterior, bounded decoded movement and hand preservation; no clean reference",
        "selection": "Final update only; whole-sequence and fit-excluded checks determine acceptance"}
