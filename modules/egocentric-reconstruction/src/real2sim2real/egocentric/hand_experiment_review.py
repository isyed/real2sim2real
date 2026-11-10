"""Paired hand metrics, calibrated overlays and editable held-out visual anchors."""

import html
import json
from pathlib import Path

import cv2
import numpy as np

from real2sim2real.artifacts import digest, write_json
from .camera import transform
from .hand_landmarks import HAND_EDGES
from .hand_review_inputs import annotation_errors
from .hand_temporal import temporal_measurements
from .interaction_io import Reconstruction, fresh_directory
from .metric_hands import project


def distribution(values):
    values = np.asarray([v for v in values if v is not None], float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, median=None, p90=None, maximum=None)
    return dict(
        count=len(values),
        median=float(np.median(values)),
        p90=float(np.quantile(values, 0.9)),
        maximum=float(values.max()),
    )


def paired_temporal(variants, times):
    """Use identical accepted edges/triples when comparing the two new models."""
    common = (
        variants["hamer"]["geometry_supported"]
        & variants["hawor"]["geometry_supported"]
    )
    result = {}
    for name in ["hamer", "hawor"]:
        velocity, acceleration, edges, triples = temporal_measurements(
            variants[name]["world_joints"], common, times
        )
        hands = []
        for h in range(2):
            norm = np.linalg.norm(acceleration[:, h], axis=-1)[triples[:, h]]
            hands.append(
                dict(
                    shared_edges=int(edges[:, h].sum()),
                    shared_triples=int(triples[:, h].sum()),
                    acceleration_rms_m_s2=float(np.sqrt(np.mean(norm**2)))
                    if norm.size
                    else None,
                    acceleration_m_s2=distribution(norm.ravel()),
                    wrist_speed_m_s=distribution(
                        np.linalg.norm(velocity[:, h, 0], axis=-1)[edges[:, h]]
                    ),
                )
            )
        result[name] = hands
    return result


def draw_projection(
    rgb, vertices, joints, faces, k, camera_from_target, supported, *, labels=()
):
    image = rgb.copy()
    for h in range(2):
        if not np.isfinite(joints[h]).all():
            continue
        v = transform(vertices[h], camera_from_target)
        j = transform(joints[h], camera_from_target)
        if (v[:, 2] <= 0).any() or (j[:, 2] <= 0).any():
            continue
        color = (0, 220, 255) if h == 0 else (255, 180, 30)
        if not supported[h]:
            color = (255, 80, 80)
        vu = np.rint(np.clip(project(v, k), -10000, 10000)).astype(np.int32)
        ju = np.rint(np.clip(project(j, k), -10000, 10000)).astype(np.int32)
        layer = image.copy()
        for face in faces[h]:
            cv2.polylines(layer, [vu[face]], True, color, 1, cv2.LINE_AA)
        image = cv2.addWeighted(image, 0.7, layer, 0.3, 0)
        for a, b in HAND_EDGES:
            cv2.line(image, tuple(ju[a]), tuple(ju[b]), color, 2, cv2.LINE_AA)
        for point in ju:
            cv2.circle(image, tuple(point), 2, color, -1, cv2.LINE_AA)
    for point in labels:
        xy = tuple(np.rint(point["xy_px"]).astype(int))
        cv2.drawMarker(image, xy, (255, 0, 255), cv2.MARKER_CROSS, 14, 2)
    return image


def annotation_editor(root, annotations, views):
    payload = json.dumps(dict(annotations=annotations, views=views)).replace(
        "<", "\\u003c"
    )
    page = """<!doctype html><meta charset="utf-8"><title>Visual hand anchors</title>
<style>body{font:16px system-ui;background:#17202b;color:#eee;max-width:1300px;margin:24px auto}canvas{width:100%;height:auto}select,input,button{padding:8px;margin:5px}p{max-width:1000px}</style>
<h1>Review visible hand landmarks</h1><p>These sparse annotations are for evaluation only. Click a visible joint to place or correct it. Leave occluded or ambiguous joints unlabelled. The initial annotations were visually estimated by an assistant and are not human-verified.</p>
<select id="view"></select><select id="frame"></select><select id="hand"><option value="0">Left hand</option><option value="1">Right hand</option></select>
<select id="joint"></select><button id="remove">Remove selected point</button><input id="reviewer" placeholder="Reviewer name"><label><input type="checkbox" id="verified">I reviewed every retained point</label><button id="save">Download annotations.json</button><canvas id="canvas"></canvas>
<script>const data=PAYLOAD;const views=data.views;const labels=data.annotations;const view=document.getElementById('view'),frame=document.getElementById('frame'),hand=document.getElementById('hand'),joint=document.getElementById('joint'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');let im=new Image();
for(const name of Object.keys(views))view.add(new Option(name,name));view.value='target';
const names=['wrist','thumb CMC','thumb MCP','thumb IP','thumb tip','index MCP','index PIP','index DIP','index tip','middle MCP','middle PIP','middle DIP','middle tip','ring MCP','ring PIP','ring DIP','ring tip','little MCP','little PIP','little DIP','little tip'];names.forEach((n,i)=>joint.add(new Option(i+': '+n,i)));joint.value=4;
function matches(p){return p.view===view.value&&p.frame_id===Number(frame.value)&&p.hand===Number(hand.value)&&p.joint===Number(joint.value)}
function draw(){ctx.drawImage(im,0,0);for(const p of labels.points.filter(p=>p.view===view.value&&p.frame_id===Number(frame.value))){ctx.strokeStyle='#ff00ff';ctx.beginPath();ctx.moveTo(p.xy_px[0]-7,p.xy_px[1]);ctx.lineTo(p.xy_px[0]+7,p.xy_px[1]);ctx.moveTo(p.xy_px[0],p.xy_px[1]-7);ctx.lineTo(p.xy_px[0],p.xy_px[1]+7);ctx.stroke();ctx.fillStyle='#ff00ff';ctx.fillText((p.hand?'R':'L')+p.joint,p.xy_px[0]+8,p.xy_px[1]-8)}}
function load(){im.onload=()=>{canvas.width=im.width;canvas.height=im.height;draw()};im.src='../evaluation_inputs/'+views[view.value].images[frame.value].path}
function frames(){frame.innerHTML='';for(const n of Object.keys(views[view.value].images))frame.add(new Option(n,n));load()};view.onchange=frames;frame.onchange=load;
canvas.onclick=e=>{const r=canvas.getBoundingClientRect();labels.points=labels.points.filter(p=>!matches(p));labels.points.push({view:view.value,frame_id:Number(frame.value),hand:Number(hand.value),joint:Number(joint.value),xy_px:[Math.round((e.clientX-r.left)*canvas.width/r.width),Math.round((e.clientY-r.top)*canvas.height/r.height)],visibility:'visible',uncertainty_px:5,image_sha256:views[view.value].images[frame.value].sha256});document.getElementById('verified').checked=false;draw()};
document.getElementById('remove').onclick=()=>{labels.points=labels.points.filter(p=>!matches(p));document.getElementById('verified').checked=false;draw()};
document.getElementById('save').onclick=()=>{const author=document.getElementById('reviewer').value.trim();labels.human_verified=Boolean(author&&document.getElementById('verified').checked);labels.reviewer=author||labels.reviewer;labels.annotation_method='Visual review in the standalone annotation editor';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(labels,null,2)],{type:'application/json'}));a.download='annotations.json';a.click()};frames();</script>"""
    (root / "annotations.html").write_text(page.replace("PAYLOAD", payload))


def evaluate(config):
    root = Path(config["output"])
    source = Reconstruction(
        config["reconstruction"], config["frame_count"], start=config["frame_start"]
    )
    view_data = json.loads((root / "evaluation_inputs/views.json").read_text())
    views = view_data["views"]
    annotations = (
        json.loads(Path(config["annotations"]).read_text())
        if config.get("annotations")
        else {"points": [], "human_verified": False}
    )
    for view in views.values():
        for image in view["images"].values():
            if digest(root / "evaluation_inputs" / image["path"]) != image["sha256"]:
                raise ValueError("A visual review image has changed")
    out = fresh_directory(root / config.get("review_directory", "review"))
    archived = np.load(root / "inputs/archived_baseline.npz")
    variants = {"archived": dict(archived)}
    summary = {}
    data_by_variant = {}
    raw_by_variant = {}
    for name in ["baseline", "hamer", "hawor"]:
        quality = json.loads((root / name / "hands_metric/quality.json").read_text())
        with np.load(root / name / "hands_metric/hypotheses.npz") as data:
            variants[name] = dict(data)
        with np.load(root / name / "hands_raw/predictions.npz") as data:
            raw_by_variant[name] = dict(data)
        data_by_variant[name] = quality
    for name, data in variants.items():
        if not np.array_equal(data["frame_ids"], source.frame_ids) or not np.allclose(
            data["timestamps_s"], source.times, atol=1e-8, rtol=0
        ):
            raise ValueError("Comparison frame IDs or capture timestamps differ")
        supported = (
            data["accepted"] if name == "archived" else data["geometry_supported"]
        )
        errors = annotation_errors(
            annotations, data["camera_joints"], source.frame_ids, views
        )
        per_hand = []
        for h in range(2):
            rows = (
                []
                if name == "archived"
                else [r[h] for r in data_by_variant[name]["records"]]
            )
            stats = {
                key: distribution([r.get(key) for r in rows])
                for key in [
                    "reprojection_median_px",
                    "validation_surface_p90_m",
                    "silhouette_precision",
                    "silhouette_iou",
                    "wrist_speed_m_s",
                ]
            }
            per_hand.append(
                dict(
                    geometry_supported_frames=int(supported[:, h].sum()),
                    recovered_vs_archived=source.frame_ids[
                        supported[:, h] & ~archived["accepted"][:, h]
                    ].tolist(),
                    lost_vs_archived=source.frame_ids[
                        ~supported[:, h] & archived["accepted"][:, h]
                    ].tolist(),
                    visual_anchor_errors={
                        v: distribution(
                            [
                                e["error_px"]
                                for e in errors
                                if e["hand"] == h and e["view"] == v
                            ]
                        )
                        for v in views
                    },
                    automatic_landmark_validated_frames=sum(
                        r.get("landmark_validation_available", False) for r in rows
                    ),
                    table_penetration=distribution(
                        [r.get("table", {}).get("penetration_depth_m") for r in rows]
                    ),
                    object_penetration=distribution(
                        [r.get("object", {}).get("penetration_depth_m") for r in rows]
                    ),
                    **stats,
                )
            )
        velocity, acceleration, edges, triples = temporal_measurements(
            data["world_joints"], supported, source.times
        )
        summary[name] = dict(
            hands=per_hand,
            visual_anchor_rows=errors,
            supported_wrist_speed_m_s=distribution(
                np.linalg.norm(velocity[..., 0, :], axis=-1)[edges]
            ),
            supported_joint_acceleration_m_s2=distribution(
                np.linalg.norm(acceleration, axis=-1)[triples].ravel()
            ),
        )
    a, b = raw_by_variant["hamer"], raw_by_variant["hawor"]
    identical = np.array_equal(a["crop_sha256"], b["crop_sha256"])
    if not identical or not np.array_equal(a["accepted"], b["accepted"]):
        raise ValueError("HaMeR and HaWoR must consume identical crops and frames")
    result = dict(
        schema="real2sim2real.hand-experiment-comparison/1",
        frame_ids=source.frame_ids.tolist(),
        variants=summary,
        identical_hamer_hawor_crops=identical,
        paired_temporal=paired_temporal(variants, source.times),
        annotations_reviewer=annotations.get("reviewer"),
        annotations_human_verified=annotations.get("human_verified", False),
        annotation_scope=annotations.get("limitations"),
        annotations_sha256=digest(config["annotations"])
        if config.get("annotations")
        else None,
        promoted=False,
        control_ready=False,
        evaluation_scope="geometry consistency and sparse visible visual anchors; no complete joint ground truth",
        comparison_note="baseline is original detector-gated raw HaMeR re-fitted with the common fitter; archived is the current refined accepted reconstruction",
    )
    write_json(out / "comparison.json", result)
    display = ["archived", "hamer", "hawor"]
    for name in display:
        (out / name).mkdir()
    for i, frame_id in enumerate(source.frame_ids):
        rgb = source.rgb(i)
        labels = [
            p
            for p in annotations.get("points", [])
            if p["frame_id"] == frame_id and p["view"] == "target"
        ]
        for name in display:
            data = variants[name]
            supported = (
                data["accepted"] if name == "archived" else data["geometry_supported"]
            )
            overlay = draw_projection(
                rgb,
                data["camera_vertices"][i],
                data["camera_joints"][i],
                data["faces"],
                source.k,
                np.eye(4),
                supported[i],
                labels=labels,
            )
            cv2.imwrite(
                str(out / name / f"target_{frame_id:06d}.jpg"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )
    for view_name, view in views.items():
        if view_name == "target":
            continue
        for frame, image in view["images"].items():
            i = int(np.flatnonzero(source.frame_ids == int(frame))[0])
            rgb = cv2.cvtColor(
                cv2.imread(str(root / "evaluation_inputs" / image["path"])),
                cv2.COLOR_BGR2RGB,
            )
            labels = [
                p
                for p in annotations.get("points", [])
                if p["frame_id"] == int(frame) and p["view"] == view_name
            ]
            for name in display:
                data = variants[name]
                supported = (
                    data["accepted"]
                    if name == "archived"
                    else data["geometry_supported"]
                )
                overlay = draw_projection(
                    rgb,
                    data["camera_vertices"][i],
                    data["camera_joints"][i],
                    data["faces"],
                    np.asarray(view["intrinsics"]),
                    np.asarray(view["view_from_target"]),
                    supported[i],
                    labels=labels,
                )
                cv2.imwrite(
                    str(out / name / f"{view_name}_{int(frame):06d}.jpg"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
    annotation_editor(out, annotations, views)
    table = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{s['hands'][0]['geometry_supported_frames']}</td><td>{s['hands'][1]['geometry_supported_frames']}</td></tr>"
        for name, s in summary.items()
    )
    payload = json.dumps(
        dict(
            ids=source.frame_ids.tolist(),
            views={k: list(v["images"]) for k, v in views.items()},
        )
    )
    page = """<!doctype html><meta charset="utf-8"><title>Hand observation comparison</title>
<style>body{font:16px system-ui;background:#111923;color:#eee;margin:24px}a{color:#80c9ff}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #465}input{width:65%}select,button{padding:8px;margin:8px}p{max-width:1100px}</style>
<h1>Hand observations in the fixed metric world</h1><p>Identical detector crops for HaMeR and HaWoR. Camera, object, table, hand shape and scale are fixed. Colours identify left/right hands; red marks a hypothesis rejected by the geometry gates. Magenta crosses are sparse visual annotations. Hidden fingers remain uncertain.</p>
<p><strong>No candidate has been promoted. This is not a control-ready reference.</strong> Initial visual anchors were reviewed by an assistant and require human confirmation. <a href="annotations.html">Review/correct annotations</a> · <a href="comparison.json">Full metrics</a></p>
<table><tr><th>Configuration</th><th>Left supported frames</th><th>Right supported frames</th></tr>TABLE</table>
<p>“archived” preserves the current refined reconstruction. “baseline” re-evaluates its original detector-gated raw predictions with the same fitter used for both new candidates. Support counts measure geometry consistency, not complete observation of every joint.</p>
<select id="view"><option>target</option></select><button id="play">Play/pause</button><input id="slider" type="range" min="0" max="45" value="20"><span id="label"></span>
<div class="grid"><div><h3>Current archived baseline</h3><img id="archived"></div><div><h3>HaMeR + shared crops</h3><img id="hamer"></div><div><h3>HaWoR + shared crops</h3><img id="hawor"></div></div>
<script>const data=PAYLOAD,slider=document.getElementById('slider'),view=document.getElementById('view');let ids=data.ids,playing=false;for(const v of Object.keys(data.views))if(v!=='target')view.add(new Option(v,v));
function update(){const f=ids[Number(slider.value)];document.getElementById('label').textContent='Source frame '+f;for(const name of ['archived','hamer','hawor'])document.getElementById(name).src=name+'/'+view.value+'_'+String(f).padStart(6,'0')+'.jpg'}
view.onchange=()=>{ids=view.value==='target'?data.ids:data.views[view.value].map(Number);slider.max=ids.length-1;slider.value=0;update()};slider.max=ids.length-1;slider.value=Math.min(20,ids.length-1);slider.oninput=update;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(Number(slider.value)+1)%ids.length;update()}},100);update();</script>"""
    (out / "review.html").write_text(
        page.replace("TABLE", table).replace("PAYLOAD", payload)
    )
    return result
