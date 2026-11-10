"""CPU review of saved grasp states; never reruns dynamics for a video."""

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from real2sim2real.artifacts import digest, write_json
from .object_grasp_pilot import load_scene, set_wrist_frame


def display_pieces(model, *, scene_only=False):
    import mujoco

    cache, pieces = {}, []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not (model.geom_group[g] == 1 or name == "table"):
            continue
        if scene_only and name not in ("table", "object_visual"):
            continue
        mesh_id = model.geom_dataid[g]
        if mesh_id < 0:
            continue
        if mesh_id not in cache:
            va, fa = model.mesh_vertadr[mesh_id], model.mesh_faceadr[mesh_id]
            mesh = trimesh.Trimesh(
                model.mesh_vert[va : va + model.mesh_vertnum[mesh_id]],
                model.mesh_face[fa : fa + model.mesh_facenum[mesh_id]],
                process=True,
            )
            budget = 2500 if name == "object_visual" else 200
            if len(mesh.faces) > budget:
                source = mesh
                for faces in (
                    [2500, 6000, 12000, 24000] if name == "object_visual" else [budget]
                ):
                    candidate = source.simplify_quadric_decimation(
                        face_count=min(faces, len(source.faces))
                    )
                    if (
                        name != "object_visual"
                        or candidate.is_watertight == source.is_watertight
                    ):
                        mesh = candidate
                        break
            cache[mesh_id] = mesh.triangles.copy()
        color = (
            np.array([175, 185, 205])
            if name != "object_visual"
            else np.array([235, 235, 235])
        )
        if name == "table":
            color = np.array([155, 104, 60])
        if "elastomer" in name:
            color = np.array([45, 165, 100])
        pieces.append((g, cache[mesh_id], color))
    return pieces


def world_triangles(data, pieces):
    triangles = []
    colors = []
    for geom, local, color in pieces:
        triangles.append(
            local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
        colors.append(np.tile(color, (len(local), 1)))
    return np.concatenate(triangles), np.concatenate(colors)


def panel(triangles, colors, azimuth, elevation, center, scale, *, depth_buffer=False):
    az, el = np.deg2rad([azimuth, elevation])
    view = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0])
    camera = np.stack([right, np.cross(view, right), view])
    camera_tri = (triangles - center) @ camera.T
    xy = camera_tri[:, :, :2] * [scale, -scale] + [260, 235]
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    shade = np.clip(
        colors * (0.5 + 0.5 * np.abs(normal @ np.array([0.3, -0.4, 0.866])))[:, None],
        0,
        255,
    ).astype(np.uint8)
    image = Image.new("RGB", (520, 460), (231, 236, 243))
    if depth_buffer:
        # Thin geometry near a large table triangle cannot be ordered correctly
        # by triangle centroids. Use orthographic per-pixel depth for plan views.
        pixels = np.asarray(image).copy()
        depth = np.full((460, 520), -np.inf)
        for t, z, color in zip(xy, camera_tri[:, :, 2], shade):
            lo = np.maximum(np.floor(t.min(0)).astype(int), [0, 0])
            hi = np.minimum(np.ceil(t.max(0)).astype(int), [519, 459])
            if (hi < lo).any():
                continue
            a, b, c = t
            denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denom) < 1e-10:
                continue
            y, x = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            x, y = x + 0.5, y + 0.5
            u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denom
            v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denom
            w = 1 - u - v
            values = u * z[0] + v * z[1] + w * z[2]
            region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            keep = (u >= 0) & (v >= 0) & (w >= 0) & (values > depth[region])
            depth[region][keep] = values[keep]
            pixels[region][keep] = color
        return Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    polygons = xy.reshape(-1, 6).tolist()
    for i in np.argsort(camera_tri[:, :, 2].mean(1)):
        draw.polygon(polygons[i], fill=tuple(shade[i]))
    return image


def render(config, directory, output, *, stem=None):
    import mujoco
    import imageio_ffmpeg

    stem = directory.name if stem is None else stem
    if not stem or Path(stem).name != stem or stem in (".", ".."):
        raise ValueError("Local nonempty review media name required")
    report = json.loads((directory / "result.json").read_text())
    if digest(directory / "states.npz") != report["states_sha256"]:
        raise ValueError("Recorded trial changed")
    if (
        digest(Path(config["output"]) / "scene/scene.xml")
        != report["source_scene_sha256"]
    ):
        raise ValueError("Recorded trial belongs to another scene")
    states = dict(np.load(directory / "states.npz"))
    model, data, _ = load_scene(config)
    candidate = report["candidate"]
    set_wrist_frame(
        model, data, config, candidate["base_position_m"], candidate["base_rotation"]
    )
    if "support" in candidate:
        from .object_grasp_pilot import set_root_frame

        hand = candidate["support"]
        set_root_frame(
            model,
            data,
            hand["root_body"],
            hand["fixture_prefix"],
            hand["base_position_m"],
            hand["base_rotation"],
        )
    pieces = display_pieces(model)
    sample = []
    for i in np.linspace(0, len(states["qpos"]) - 1, 8).astype(int):
        data.qpos[:] = states["qpos"][i]
        mujoco.mj_forward(model, data)
        sample.append(world_triangles(data, pieces)[0].reshape(-1, 3))
    bounds = np.array([np.concatenate(sample).min(0), np.concatenate(sample).max(0)])
    center = bounds.mean(0)
    center[2] -= 0.015
    scale = min(850.0, 440.0 / max(np.ptp(bounds, axis=0)))
    count = len(states["qpos"])
    step = 2
    fps = 1 / float(np.median(np.diff(states["timestamps_s"]))) / step
    path = output / (stem + ".mp4")
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (1040, 520),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        output_params=[
            "-threads",
            "1",
            "-preset",
            "veryfast",
            "-movflags",
            "+faststart",
        ],
    )
    writer.send(None)
    indices = list(range(0, count, step))
    try:
        for i in indices:
            data.qpos[:] = states["qpos"][i]
            mujoco.mj_forward(model, data)
            triangles, colors = world_triangles(data, pieces)
            image = Image.new("RGB", (1040, 520), (248, 250, 252))
            draw = ImageDraw.Draw(image)
            image.paste(panel(triangles, colors, -60, 28, center, scale), (0, 40))
            image.paste(panel(triangles, colors, 125, 55, center, scale), (520, 40))
            row = report["records"][i]
            draw.text(
                (15, 10),
                f"Saved physics | {stem} | {row['phase']} | {row['time_s']:.2f}s",
                fill=(20, 30, 45),
            )
            draw.text(
                (15, 502),
                f"Object bottom {row['minimum_object_height_m'] * 1000:.1f} mm | wrist fixture, arm not modelled | no object assistance",
                fill=(20, 30, 45),
            )
            writer.send(np.asarray(image))
            if i == indices[len(indices) // 2] or i == indices[-1]:
                image.save(output / (stem + f"-{i:04d}.jpg"), quality=92)
    finally:
        writer.close()
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    decoded = sum(1 for _ in reader)
    reader.close()
    if decoded != len(indices):
        raise ValueError("Rendered video incomplete")
    from real2sim2real.review_media import prepare

    portable = prepare(path, output / (stem + ".playback"))
    return dict(
        path=path.name,
        frames=decoded,
        fps=fps,
        size=meta["size"],
        sha256=digest(path),
        state_sha256=report["states_sha256"],
        display_only_decimation=True,
        dynamics_rerun=False,
        portable_media=portable,
    )


def review(config):
    import mujoco
    from real2sim2real.review_media import player

    root = Path(config["output"])
    output = root / config.get("review_directory", "review")
    output.mkdir(exist_ok=False)
    evaluation_root = root / config.get("evaluation_directory", "evaluation")
    evaluation = json.loads((evaluation_root / "evaluation.json").read_text())
    videos = []
    selected = evaluation.get("selected_trial")
    nominal = evaluation["trials"]
    chosen = next((r for r in nominal if r["directory"] == selected), None)
    if chosen is None and nominal:
        chosen = max(nominal, key=lambda r: r["hold_minimum_clearance_m"])
    render_rows = ([] if chosen is None else [chosen]) + evaluation["variants"]
    failed = next((r for r in nominal if not r["success"] and r is not chosen), None)
    if failed is not None:
        render_rows.append(failed)
    for row in render_rows:
        videos.append(render(config, evaluation_root / row["directory"], output))
    model, data, _ = load_scene(config)
    pieces = display_pieces(model, scene_only=True)
    settled = json.loads((root / "settling.json").read_text())
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task_object_free")
    qid = model.jnt_qposadr[j]
    initial = Image.new("RGB", (1040, 500), (248, 250, 252))
    draw = ImageDraw.Draw(initial)
    for col, label in enumerate(
        ["Reconstructed initial pose", "Gravity-settled task start"]
    ):
        if col:
            data.qpos[qid : qid + 7] = settled["object_qpos"]
        mujoco.mj_forward(model, data)
        triangles, colors = world_triangles(data, pieces)
        initial.paste(
            panel(triangles, colors, -60, 25, np.array([0.025, 0.065, 0.04]), 850),
            (col * 520, 40),
        )
        draw.text((col * 520 + 15, 10), label, fill=(20, 30, 45))
    initial.save(output / "initialization.jpg", quality=95)
    rows = []
    for row in [*evaluation["trials"], *evaluation["variants"]]:
        rows.append(
            f"<tr><td>{html.escape(row['directory'])}</td><td>{row['success']}</td><td>{row['hold_minimum_clearance_m'] * 1000:.1f}</td><td>{row['max_robot_penetration_m'] * 1000:.2f}</td><td>{html.escape(', '.join(row['rejected_for']))}</td></tr>"
        )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Object-centric grasp pilot</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto;background:#18202b;color:#eee}a{color:#8cf}video,img{width:100%}td,th{padding:8px;text-align:left}pre{white-space:pre-wrap}</style><h1>Object-centric grasp and lift</h1><p>Actual saved CPU MuJoCo states. Force-limited wrist fixture and articulated Sharpa hand; arm reachability and hardware execution are not validated. Human finger trajectories are not targets. The object has no actuator, weld, external force or gravity assistance.</p><h2>'
        + html.escape(evaluation["decision"])
        + "</h2><table><tr><th>Trial</th><th>Pass</th><th>Hold clearance, mm</th><th>Max robot penetration, mm</th><th>Failures</th></tr>"
        + "".join(rows)
        + "</table><h2>Starting state</h2><p>The gravity-settled task start differs from the reconstructed pose by "
        + f"{settled['translation_m'] * 1000:.1f} mm and {settled['rotation_deg']:.1f} degrees. The original reconstruction is preserved."
        + '</p><img src="initialization.jpg">'
        + "".join(
            f"<h2>{html.escape(v['path'])}</h2>"
            + player(
                v["path"],
                Path(v["path"]).stem + ".playback/video.webm",
                Path(v["path"]).stem + ".playback/preview.gif",
                embedded_gif=output / (Path(v["path"]).stem + ".playback/preview.gif"),
                embedded_webm=output / (Path(v["path"]).stem + ".playback/video.webm"),
            )
            for v in videos
        )
        + f'<p><a href="../{evaluation_root.name}/evaluation.json">Evaluation</a> | <a href="../scene/collision_audit.json">Collision proxy audit</a> | <a href="../settling.json">Settling</a></p>'
    )
    write_json(
        output / "review.json",
        dict(
            videos=videos,
            evaluation_sha256=digest(evaluation_root / "evaluation.json"),
            config=config,
        ),
    )
    return videos


def unique_trial_media_names(trials):
    """Preserve unique stems and disambiguate repeated candidate names by study."""
    paths = [Path(p).resolve() for p in trials]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate review trial paths would overwrite media")
    depth = 1
    names = [p.name for p in paths]
    while len(set(names)) != len(names):
        depth += 1
        names = ["--".join(p.parts[-depth:]) for p in paths]
    return names


def review_trials(
    config, trials, output, *, title="Fixed-grasp controller diagnostics"
):
    """Compare named saved trials across bounded studies without rerunning them."""
    from real2sim2real.review_media import player

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sections, videos = [], []
    trials = list(map(Path, trials))
    for directory, stem in zip(trials, unique_trial_media_names(trials)):
        result = json.loads((directory / "result.json").read_text())
        video = render(config, directory, output, stem=stem)
        videos.append(
            dict(
                **video,
                source_result=str(directory / "result.json"),
                source_result_sha256=digest(directory / "result.json"),
            )
        )
        verdict = (
            "Physical gates passed"
            if result["success"]
            else "Physical gates rejected: " + ", ".join(result["rejected_for"])
        )
        motion_condition = (
            "Demonstrated object-motion increments"
            if result.get("configuration", {}).get("reference_guidance")
            else "Stationary lift/hold diagnostic"
        )
        support_note = f"<p>Motion condition: {motion_condition}.</p>"
        protocol_path = directory.parent / "protocol.json"
        if (
            result.get("configuration", {}).get("reference_guidance")
            and protocol_path.exists()
        ):
            protocol = json.loads(protocol_path.read_text())
            limits = protocol.get("fidelity_limits")
            if limits:
                from .retention_experiment import assess

                assessment = assess(result, limits)
                score = assessment["fidelity"]
                support_note += f"<p>Complete physical plus reference acceptance: {'pass' if assessment['combined_pass'] else 'fail'}.</p>"
                if score:
                    support_note += (
                        f"<p>Peak reference position error {score['maximum_position_error_m'] * 1000:.2f} mm "
                        f"(limit {limits['position_m'] * 1000:g} mm); rotation error "
                        f"{score['maximum_rotation_error_deg']:.2f} degrees (limit {limits['rotation_deg']:g}).</p>"
                    )
        if "support_load_fraction" in result:
            drift = result["support_wrist_drift_m"]
            support_note += (
                f"<p>Second hand: mean upward force on the object {result['support_mean_upward_force_n']:.2f} N; "
                f"at least 10% of object weight in {result['support_load_fraction'] * 100:.1f}% of hold samples; "
                f"support-wrist drift {'unavailable' if drift is None else f'{drift * 1000:.1f} mm'}. "
                f"Support enabled: {result['support_controller'].get('enabled', True)}. "
                "Both wrists are force limited; no arms are modelled.</p>"
            )
        from .wrist_coordination import hold_world_motion

        world_drift = hold_world_motion(result["records"])[
            "hold_world_horizontal_drift_m"
        ]
        hold = [r for r in result["records"] if r["phase"] == "hold"]
        rotation = max((r["rotation_change_deg"] for r in hold), default=None)
        motion_note = (
            "<p>Independent world-frame horizontal drift: "
            + ("unavailable" if world_drift is None else f"{world_drift * 1000:.2f} mm")
            + "; maximum object rotation: "
            + ("unavailable" if rotation is None else f"{rotation:.2f} degrees")
            + f"; opposing-finger contact in {result['hold_opposed_contact_fraction'] * 100:.1f}% of hold samples.</p>"
        )
        contact_note = ""
        if result.get("hold_contact_support"):
            contact_note = "<table><tr><th>Primary contact group</th><th>Mean normal force (N)</th><th>Active hold samples</th></tr>"
            for group, evidence in result["hold_contact_support"].items():
                contact_note += (
                    f"<tr><td>{html.escape(group)}</td><td>{evidence['mean_normal_force_n']:.3f}</td>"
                    f"<td>{evidence['active_sample_fraction'] * 100:.1f}%</td></tr>"
                )
            contact_note += "</table><p>Active means normal force above 0.1 N. Normal force is not vertical load share.</p>"
        if result.get("closure_target_geometry"):
            error = max(
                result["closure_target_geometry"]["closed_command_target_error_m"]
            )
            contact_note += (
                f"<p>Largest unconstrained closed-command fingertip error: {error * 1000:.1f} mm. "
                "This kinematic command audit is separate from actual loaded contacts above.</p>"
            )
        if "coordination" in result:
            diag = result["coordination"]
            motion_note += (
                "<p>Common wrist correction active; joint-limit projection used in "
                f"{diag.get('coordination_projected_steps', 'unrecorded')} of {diag.get('coordination_update_steps', 'unrecorded')} updates. "
                "Exact simulated object pose is a privileged feedback input. "
                "Reduced drift alone does not establish stable contact.</p>"
            )
        sections.append(
            f"<h2>{html.escape(stem)}</h2><p>{html.escape(verdict)}. "
            f"Mass {config['physics']['mass_kg'] * result['mass_scale']:.2f} kg; friction {result['friction']:.2f}. "
            f"Minimum hold clearance {result['hold_minimum_clearance_m'] * 1000:.1f} mm; "
            f"hold drift {result['hold_slip_m'] * 1000:.1f} mm ({html.escape(result.get('hold_slip_frame', 'world-vector legacy metric'))} frame).</p>"
            + support_note
            + motion_note
            + contact_note
            + player(
                video["path"],
                stem + ".playback/video.webm",
                stem + ".playback/preview.gif",
                embedded_gif=output / (stem + ".playback/preview.gif"),
                embedded_webm=output / (stem + ".playback/video.webm"),
            )
        )
    (output / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>body{{font:17px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#18202b;color:#eee}}a{{color:#8cf}}video,img{{width:100%}}details{{padding:12px;background:#293745}}</style>"
        f"<h1>{html.escape(title)}</h1><p>Actual saved CPU physics. Articulated hand and force-limited wrist fixture; no arm or object assistance. Human finger trajectories are not targets. Short passes do not erase longer hold failures.</p>"
        + "".join(sections)
        + "</html>"
    )
    write_json(
        output / "review.json", dict(videos=videos, config=config, physics_rerun=False)
    )
    return videos
