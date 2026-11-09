"""Standalone, offline scene review with a frame slider and rotatable 3D view."""

import json
from pathlib import Path

import cv2
import numpy as np

from real2sim2real.artifacts import write_json
from .camera import transform
from .hand_landmarks import HAND_EDGES, draw_hand
from .interaction_io import Reconstruction, fresh_directory
from .metric_hands import project


def review(config):
    import trimesh

    source = Reconstruction(config["reconstruction"], config.get("frame_count"))
    root = Path(config["output"])
    hand_path = root / "hands_metric/hands.npz"
    out = fresh_directory(root / ("review" if hand_path.exists() else "review-preview"))
    (out / "overlays").mkdir()
    support = json.loads((root / "support/support.json").read_text())
    scene = np.load(root / "support/scene.npz")
    contacts_path = (
        root
        / ("contacts" if hand_path.exists() else "contacts-preview")
        / "diagnostics.json"
    )
    contacts = json.loads(contacts_path.read_text()) if contacts_path.exists() else None
    metrics_path = root / "quality_summary.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    comparison_path = root / "hand_comparison.json"
    comparison = (
        json.loads(comparison_path.read_text()) if comparison_path.exists() else None
    )
    hands = np.load(hand_path) if hand_path.exists() else None
    hand_quality_path = root / "hands_metric/quality.json"
    hand_quality = (
        json.loads(hand_quality_path.read_text())
        if hand_quality_path.exists()
        else None
    )
    observations_path = root / "hands2d/landmarks.npz"
    observations = np.load(observations_path) if observations_path.exists() else None
    mesh = trimesh.load(source.path("mesh/metric.glb"), force="mesh")
    sample = np.random.default_rng(0).choice(
        len(mesh.vertices), min(3500, len(mesh.vertices)), replace=False
    )
    boundary = np.c_[
        support["observed_boundary_xy"], np.zeros(len(support["observed_boundary_xy"]))
    ]
    frames = []
    for i in range(source.count):
        image = cv2.cvtColor(source.rgb(i), cv2.COLOR_RGB2BGR)
        uv = project(
            transform(boundary, np.linalg.inv(scene["world_from_camera"][i])), source.k
        )
        cv2.polylines(
            image, [np.rint(uv).astype(np.int32)], True, (170, 220, 100), 2, cv2.LINE_AA
        )
        hand_joints = [None, None]
        hand_vertices = [None, None]
        for h, color in enumerate([(255, 160, 30), (40, 190, 255)]):
            if hands is not None and hands["accepted"][i, h]:
                draw_hand(image, project(hands["camera_joints"][i, h], source.k), color)
                hand_joints[h] = hands["world_joints"][i, h].round(5).tolist()
                hand_vertices[h] = hands["world_vertices"][i, h, ::3].round(5).tolist()
            elif observations is not None and observations["accepted"][i, h]:
                draw_hand(image, observations["keypoints_px"][i, h], (150, 150, 150))
        label = "Metric 3D hands: " + (
            str(hands["accepted"][i].tolist())
            if hands is not None
            else "NOT AVAILABLE; grey is 2D only"
        )
        cv2.rectangle(image, (0, 0), (image.shape[1], 36), (25, 25, 25), -1)
        cv2.putText(
            image,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out / "overlays" / f"{i:06d}.jpg"), image)
        frames.append(
            dict(
                id=int(source.frame_ids[i]),
                time=float(source.times[i]),
                object=scene["world_from_object"][i].round(7).tolist(),
                camera=scene["world_from_camera"][i, :3, 3].round(5).tolist(),
                hands=hand_joints,
                hand_vertices=hand_vertices,
                contacts=contacts["records"][i] if contacts else None,
                fit_quality=hand_quality["records"][i] if hand_quality else None,
            )
        )
    data = dict(
        frames=frames,
        boundary=boundary.round(5).tolist(),
        object=np.asarray(mesh.vertices)[sample].round(5).tolist(),
        edges=HAND_EDGES,
        support=support,
        has_hands=hands is not None,
        summary=contacts["summary"] if contacts else None,
        metrics=metrics,
        comparison=comparison,
        fit_limits=hand_quality["limits"] if hand_quality else None,
    )
    write_json(
        out / "summary.json",
        dict(
            total_frames=source.count,
            metric_hands_available=hands is not None,
            support_accepted=support["accepted"],
            accepted_hand_frames=hands["accepted"].sum(0).tolist()
            if hands is not None
            else [0, 0],
        ),
    )
    # Data are numeric plus controlled labels. Escape closing tags defensively.
    payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    (out / "review.html").write_text(HTML.replace("__DATA__", payload))
    return out / "review.html"


HTML = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Metric interaction review</title>
<style>
body{font:15px system-ui,sans-serif;background:#111822;color:#e9eff6;margin:24px;line-height:1.5}
h1{font-size:24px;margin-bottom:8px}.note{color:#b8c7d7;max-width:1100px}.warning{color:#ffcf77}
.views{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:18px 0}img,canvas{width:100%;aspect-ratio:1014/760;background:#192331;border-radius:8px;object-fit:contain}
canvas{touch-action:none;cursor:grab}.bar{display:flex;align-items:center;gap:16px}input{flex:1}button{padding:8px 16px;background:#2b4a68;border:0;border-radius:5px;color:white}
pre{white-space:pre-wrap;background:#192331;padding:16px;border-radius:8px;font-size:13px}.key{color:#91c4e7}@media(max-width:850px){.views{grid-template-columns:1fr}}
</style><h1>Tabletop, object and hand reconstruction</h1>
<div id="status" class="warning"></div>
<p class="note">Fixed metric world; table at z = 0, +Z up. Up assumes a level tabletop. The boundary is the observed support hull. Drag the 3D view to rotate; scroll to zoom. Grey hand overlays are 2D observations without an accepted 3D estimate.</p>
<div class="bar"><button id="play">Play</button><input id="frame" aria-label="Frame" type="range" min="0" value="0"><span id="label"></span></div>
<div class="views"><div><img id="rgb" alt="Calibrated image with table and hand overlays"><div class="key">Image projection · left cyan · right orange · table green</div></div><div><canvas id="scene" width="1014" height="760" aria-label="Rotatable metric 3D scene"></canvas><div class="key">World view · object white · table green · camera violet</div></div></div>
<p class="note">Proximity indicates a contact candidate, not verified physical contact. Penetration depends on reconstructed surfaces. Missing hands remain missing. These diagnostics do not certify a control-ready demonstration.</p>
<pre id="diagnostic"></pre><script>
const DATA=__DATA__;
const slider=document.getElementById('frame'), canvas=document.getElementById('scene'), ctx=canvas.getContext('2d');
slider.max=DATA.frames.length-1;let yaw=-.3,pitch=.7,zoom=850,drag=null,playing=false,timer;
document.getElementById('status').textContent=DATA.has_hands?(DATA.metrics?`Accepted metric frames: left ${DATA.metrics.hands.left.accepted_metric_frames}/${DATA.frames.length}, right ${DATA.metrics.hands.right.accepted_metric_frames}/${DATA.frames.length}. Missing estimates remain invalid.`:'Accepted metric hand estimates are shown; inspect coverage and contact errors below.'):'PARTIAL REVIEW: tabletop and object are reconstructed. Articulated metric 3D hands are not yet available.';
if(DATA.comparison)document.getElementById('status').textContent+=` Recovered: left ${DATA.comparison.hands.left.recovered_frames}, right ${DATA.comparison.hands.right.recovered_frames}. Previously accepted geometry is unchanged.`;
function projected(p){const x=p[0],y=p[1]-.12,z=p[2]-.10,a=Math.cos(yaw)*x-Math.sin(yaw)*y,b=Math.sin(yaw)*x+Math.cos(yaw)*y;return [507+zoom*a,380-zoom*(Math.cos(pitch)*z-Math.sin(pitch)*b),Math.sin(pitch)*z+Math.cos(pitch)*b];}
function line(a,b,color,width=2){a=projected(a);b=projected(b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
function pose(p,t){return [0,1,2].map(i=>t[i][0]*p[0]+t[i][1]*p[1]+t[i][2]*p[2]+t[i][3]);}
function draw(){const f=DATA.frames[+slider.value];ctx.clearRect(0,0,1014,760);
const boundary=DATA.boundary.map(projected);ctx.fillStyle='#244837';ctx.beginPath();boundary.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fill();
DATA.boundary.forEach((p,i)=>line(p,DATA.boundary[(i+1)%DATA.boundary.length],'#92c985'));
line([0,0,0],[.12,0,0],'#f47777');line([0,0,0],[0,.12,0],'#91c985');line([0,0,0],[0,0,.12],'#7cacff');
let points=DATA.object.map(p=>({p:projected(pose(p,f.object)),c:'#e1e5e8'}));
f.hand_vertices.forEach((v,h)=>{if(v)v.forEach(p=>points.push({p:projected(p),c:h?'#efbb56':'#50b6ee'}));});
points.sort((a,b)=>a.p[2]-b.p[2]);points.forEach(p=>{ctx.fillStyle=p.c;ctx.fillRect(p.p[0],p.p[1],2.5,2.5);});
f.hands.forEach((j,h)=>{if(j)DATA.edges.forEach(e=>line(j[e[0]],j[e[1]],h?'#ffc15b':'#60cfff',3));});
const camera=projected(f.camera);ctx.fillStyle='#bd97ed';ctx.beginPath();ctx.arc(camera[0],camera[1],6,0,Math.PI*2);ctx.fill();
document.getElementById('rgb').src='overlays/'+String(+slider.value).padStart(6,'0')+'.jpg';
document.getElementById('label').textContent='Frame '+f.id+' · '+f.time.toFixed(2)+' s';
document.getElementById('diagnostic').textContent=JSON.stringify({frame:f.id,units:'metres; reprojection errors in pixels',metric_hands:f.hands.map(Boolean),fit_quality:f.fit_quality,fit_limits:DATA.fit_limits,diagnostics:f.contacts},null,2);}
slider.oninput=draw;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=()=>drag=null;
canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.007;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.007));drag=[e.clientX,e.clientY];draw();}};
canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(200,Math.min(2500,zoom*Math.exp(-e.deltaY*.001)));draw();};
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';if(!playing){clearTimeout(timer);return;}function advance(){if(!playing)return;let i=+slider.value;slider.value=(i+1)%DATA.frames.length;draw();let dt=i+1<DATA.frames.length?DATA.frames[i+1].time-DATA.frames[i].time:.05;timer=setTimeout(advance,Math.max(10,dt*1000));}advance();};draw();
</script></html>"""
