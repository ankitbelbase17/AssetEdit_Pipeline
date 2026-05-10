 """
!pip install trimesh
!pip install pyvista
!pip install open3d
Batch Mesh Uncertainty Analysis  v3
======================================
Paints the uncertainty heatmap directly onto the original mesh file
(mesh_000 of each object), preserving its exact vertices and faces.

How it works
-------------
For each object folder:
  1. Load all ~20 meshes.  mesh_000 becomes the canonical reference.
  2. ICP-align every other mesh to mesh_000.
  3. For each vertex of mesh_000, find its nearest neighbour in every
     other mesh → stack positions → compute per-vertex std/mad/range.
  4. Map scores → plasma colours → write them into mesh_000's vertex
     colour channel → export as .ply  (and optionally overwrite the
     original or write a sidecar).

This means the output mesh has EXACTLY the same vertices and faces as
mesh_000, just with a colour per vertex encoding geometric uncertainty.

Output per object
------------------
  output_root/
    chair/
      uncertainty_colored.ply   ← original mesh + heatmap colours
      pyvista_iso.png
      pyvista_views.png
      interactive_heatmap.html  ← rotate in browser
      uncertainty_stats.png
      similarity_matrix.png
      uncertainty_scores.npy    ← (N_verts,) float32
      summary.txt
    ...
    _batch_report/
      batch_summary.html
      batch_comparison.png
      batch_summary.csv

Config
-------
  ROOT_DIR   → folder whose immediate subdirs are per-object mesh folders
  MESH_GLOB  → glob pattern for mesh files inside each subfolder
"""

import os, glob, copy, warnings, time, csv
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import trimesh
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import seaborn as sns

from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

import pyvista as pv
pv.global_theme.background = "black"
pv.global_theme.font.color  = "white"

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration  ← EDIT THESE
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    ROOT_DIR           = "./root_meshes"   # parent folder of per-object subfolders
    OUTPUT_ROOT        = "./batch_output_v3"
    MESH_GLOB          = "*.obj"           # "*.ply", "*.glb", etc.

    DO_ICP             = True
    ICP_MAX_ITER       = 50
    ICP_THRESHOLD      = 0.02

    # "std" | "range" | "mad"
    UNCERTAINTY_METRIC = "std"

    COLORMAP           = "plasma"
    PLOTLY_COLORSCALE  = "Plasma"
    EXPORT_PLY         = True
    FIGURE_DPI         = 150

    SKIP_IF_DONE       = False   # set True to resume a crashed batch
    MIN_MESHES         = 2

cfg = Config()
os.makedirs(cfg.OUTPUT_ROOT, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Mesh helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_meshes(obj_dir: str) -> List[trimesh.Trimesh]:
    paths = sorted(glob.glob(os.path.join(obj_dir, cfg.MESH_GLOB)))
    meshes = []
    for p in paths:
        m = trimesh.load(p, force="mesh", process=True)
        m.metadata["source"] = os.path.basename(p)
        meshes.append(m)
    return meshes


def trimesh_to_o3d(mesh):
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(mesh.vertices)
    m.triangles = o3d.utility.Vector3iVector(mesh.faces)
    m.compute_vertex_normals()
    return m


def o3d_to_trimesh(mesh):
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                           faces=np.asarray(mesh.triangles), process=False)


def center_and_scale(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate centroid → origin, normalise to unit bounding-sphere."""
    m = mesh.copy()
    m.metadata = mesh.metadata.copy()
[3/23/2026 5:36 AM] Anish Bishwakarma: m.vert
[3/23/2026 5:36 AM] Anish Bishwakarma: etadata = mesh.metadata
[3/23/2026 5:36 AM] Anish Bishwakarma: .copy()
[3/23/2026 5:36 AM] Anish Bishwakarma: m.metadata = mesh.metadata.copy()
    m.vertices -= m.vertices.mean(axis=0)
    r = np.linalg.norm(m.vertices, axis=1).max()
    if r > 0:
        m.vertices /= r
    return m


def icp_align(source: trimesh.Trimesh,
              target: trimesh.Trimesh) -> trimesh.Trimesh:
    """ICP-align *source* to *target*; return aligned copy."""
    src_o3d = trimesh_to_o3d(source)
    tgt_o3d = trimesh_to_o3d(target)
    reg = o3d.pipelines.registration.registration_icp(
        src_o3d.sample_points_uniformly(4096),
        tgt_o3d.sample_points_uniformly(4096),
        max_correspondence_distance=cfg.ICP_THRESHOLD,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=cfg.ICP_MAX_ITER),
    )
    aligned_o3d = copy.deepcopy(src_o3d)
    aligned_o3d.transform(reg.transformation)
    out = o3d_to_trimesh(aligned_o3d)
    out.metadata = source.metadata.copy()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ★  Core: project all meshes onto the ORIGINAL reference vertices
# ══════════════════════════════════════════════════════════════════════════════

def build_stacked_on_original(
        meshes: List[trimesh.Trimesh]
) -> tuple:
    """
    Returns
    -------
    ref_mesh   : trimesh  — mesh_000 normalised to unit sphere (canonical)
    aligned    : list     — all M meshes normalised + ICP-aligned to ref_mesh
    stacked    : ndarray (M, V, 3) — nearest surface point on each aligned
                 mesh for every vertex of ref_mesh
    """
    # 1. Normalise all meshes (centre + unit sphere)
    normalised = [center_and_scale(m) for m in meshes]
    ref = normalised[0]

    # 2. ICP-align every other mesh to the reference
    aligned = [ref]
    if cfg.DO_ICP:
        for m in normalised[1:]:
            aligned.append(icp_align(m, ref))
    else:
        aligned = normalised

    # 3. Project: for each ref vertex, find nearest surface point on every mesh
    ref_verts = ref.vertices          # (V, 3)
    V = len(ref_verts)
    M = len(aligned)
    stacked = np.zeros((M, V, 3), dtype=np.float32)
    stacked[0] = ref_verts.astype(np.float32)   # mesh_000 maps to itself

    for i, mesh in enumerate(aligned[1:], 1):
        pts, _ = trimesh.sample.sample_surface(mesh, max(V * 8, 20_000))
        _, idx = cKDTree(pts).query(ref_verts, workers=-1)
        stacked[i] = pts[idx].astype(np.float32)

    return ref, aligned, stacked


# ══════════════════════════════════════════════════════════════════════════════
# Uncertainty & similarity
# ══════════════════════════════════════════════════════════════════════════════

def compute_uncertainty(stacked: np.ndarray) -> np.ndarray:
    """(M, V, 3) → (V,) uncertainty score per original vertex."""
    dist = np.linalg.norm(stacked - stacked.mean(axis=0)[np.newaxis], axis=2)
    m = cfg.UNCERTAINTY_METRIC
    if   m == "std"  : return dist.std(axis=0).astype(np.float32)
    elif m == "range": return (dist.max(axis=0)-dist.min(axis=0)).astype(np.float32)
    elif m == "mad"  :
        med = np.median(dist, axis=0)
        return np.median(np.abs(dist - med), axis=0).astype(np.float32)
    raise ValueError(m)


def per_vertex_statistics(stacked: np.ndarray) -> dict:
    dist = np.linalg.norm(stacked - stacked.mean(axis=0)[np.newaxis], axis=2)
    return {
        "mean_displacement" : dist.mean(axis=0),
        "std_displacement"  : dist.std(axis=0),
        "max_displacement"  : dist.max(axis=0),
        "range_displacement": dist.max(axis=0) - dist.min(axis=0),
        "mad_displacement"  : np.median(np.abs(dist - np.median(dist,axis=0)),axis=0),
        "mean_position"     : stacked.mean(axis=0),
    }


def chamfer_distance(A, B):
    return float(cKDTree(B).query(A, workers=-1)[0].mean() +
                 cKDTree(A).query(B, workers=-1)[0].mean())


def build_similarity_matrix(stacked: np.ndarray) -> np.ndarray:
    M = stacked.shape[0]
    mat = np.zeros((M, M), dtype=np.float32)
    for i in range(M)
[3/23/2026 5:36 AM] Anish Bishwakarma: :
        for j in range(i+1, M):
            cd = chamfer_distance(stacked[i], stacked[j])
            mat[i,j] = mat[j,i] = cd
    return mat


def chamfer_to_similarity(cd):
    d = cd.max()
    return np.ones_like(cd) if d == 0 else 1.0 - cd / d


# ══════════════════════════════════════════════════════════════════════════════
# Colour helpers
# ══════════════════════════════════════════════════════════════════════════════

def scores_to_rgba(scores: np.ndarray) -> np.ndarray:
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    return (cm.get_cmap(cfg.COLORMAP)(norm(scores)) * 255).astype(np.uint8)

def scores_to_rgb_float(scores: np.ndarray) -> np.ndarray:
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    return cm.get_cmap(cfg.COLORMAP)(norm(scores))[:, :3]

def _dark(ax):
    ax.tick_params(colors="gray", labelsize=8)
    ax.spines[:].set_edgecolor("#333")
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_color("gray")


# ══════════════════════════════════════════════════════════════════════════════
# ★  Export: paint heatmap onto the original mesh
# ══════════════════════════════════════════════════════════════════════════════

def sanitise_name(filename: str) -> str:
    """
    Turn an input filename into a clean PLY stem.
      'mesh_000.obj'       → 'mesh_000'
      'My Object 001.obj'  → 'My-Object-001'
      'scan file.ply'      → 'scan-file'
    Rules: strip extension, replace spaces with '-',
    collapse any run of non-alphanumeric (except '-' and '_') to '-'.
    """
    import re
    stem = Path(filename).stem
    stem = stem.replace(" ", "-")
    stem = re.sub(r"[^\w\-]", "-", stem)   # \w = [a-zA-Z0-9_]
    stem = re.sub(r"-{2,}", "-", stem)     # collapse consecutive hyphens
    return stem.strip("-")


def export_all_heatmaps(
        aligned:   List[trimesh.Trimesh],
        ref_mesh:  trimesh.Trimesh,
        scores:    np.ndarray,
        out_dir:   str,
) -> List[str]:
    """
    For every mesh in *aligned*, project the per-vertex uncertainty *scores*
    (defined on ref_mesh's V vertices) onto that mesh's own vertices via KNN,
    then export as a vertex-coloured PLY.

    Output filenames mirror the original source filename with spaces → '-'
    and extension swapped to '.ply'.

    Returns the list of written file paths.
    """
    ply_dir = os.path.join(out_dir, "colored_ply")
    os.makedirs(ply_dir, exist_ok=True)

    ref_verts = ref_mesh.vertices          # (V, 3) — scores live here
    norm      = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap      = cm.get_cmap(cfg.COLORMAP)
    written   = []

    for i, mesh in enumerate(aligned):
        # Project: for each vertex of this mesh, find nearest ref vertex → borrow its score
        _, idx   = cKDTree(ref_verts).query(mesh.vertices, workers=-1)
        v_scores = scores[idx]                         # (Vi,)

        rgba = (cmap(norm(v_scores)) * 255).astype(np.uint8)

        m = mesh.copy()
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)

        src_name  = mesh.metadata.get("source", f"mesh_{i:03d}.obj")
        out_name  = sanitise_name(src_name) + ".ply"
        out_path  = os.path.join(ply_dir, out_name)
        m.export(out_path)
        written.append(out_path)

    return written


# ══════════════════════════════════════════════════════════════════════════════
# PyVista render  (uses original mesh topology)
# ══════════════════════════════════════════════════════════════════════════════

def render_pyvista(ref_mesh: trimesh.Trimesh, scores: np.ndarray,
                   out_dir: str, obj_name: str):
    verts = ref_mesh.vertices.astype(np.float32)
    faces = ref_mesh.faces
    pv_faces = np.hstack([np.full((len(faces),1),3,dtype=np.int64),
                           faces.astype(np.int64)]).ravel()
    pv_mesh  = pv.PolyData(verts, pv_faces)
    pv_mesh["uncertainty"] = scores.astype(np.float64)
    clim  = [float(scores.min()), float(scores.max())]
    sargs = dict(title="Uncertainty", title_font_size=12, label_font_size=10,
[3/23/2026 5:36 AM] Anish Bishwakarma: 10,
[3/23/2026 5:36 AM] Anish Bishwakarma: t_size=12, label_font_size=10,
                 shadow=True, n_labels=5, fmt="%.4f", font_family="arial")

    # 4-panel
    pl = pv.Plotter(shape=(2,2), off_screen=True, window_size=(1400,1100))
    for idx,(title,pos,up) in enumerate([
        ("Front", (0,0,4),  (0,1,0)),
        ("Back",  (0,0,-4), (0,1,0)),
        ("Right", (4,0,0),  (0,1,0)),
        ("Top",   (0,4,0),  (0,0,-1)),
    ]):
        pl.subplot(*divmod(idx,2))
        pl.add_mesh(pv_mesh, scalars="uncertainty", cmap=cfg.COLORMAP,
                    clim=clim, smooth_shading=True, show_edges=False,
                    lighting=True, scalar_bar_args=sargs)
        pl.add_text(title, font_size=11, color="white", position="upper_left")
        pl.camera.position    = pos
        pl.camera.focal_point = (0,0,0)
        pl.camera.up          = up
        pl.camera.reset_clipping_range()
        pl.add_axes(color="white")
    pl.screenshot(os.path.join(out_dir,"pyvista_views.png"), return_img=False)
    pl.close()

    # Isometric
    pl2 = pv.Plotter(off_screen=True, window_size=(1200,1000))
    pl2.set_background("#0d0d0d")
    pl2.add_mesh(pv_mesh, scalars="uncertainty", cmap=cfg.COLORMAP,
                 clim=clim, smooth_shading=True, show_edges=False,
                 lighting=True, scalar_bar_args={**sargs,"title_font_size":16})
    pl2.add_text(f"{obj_name}  —  Uncertainty Heatmap (original mesh)",
                 font_size=13, color="white", position="upper_edge")
    pl2.camera.position=(3,2,3); pl2.camera.focal_point=(0,0,0)
    pl2.camera.up=(0,1,0); pl2.camera.reset_clipping_range()
    pl2.add_axes(color="white")
    pl2.screenshot(os.path.join(out_dir,"pyvista_iso.png"), return_img=False)
    pl2.close()


# ══════════════════════════════════════════════════════════════════════════════
# Plotly interactive HTML
# ══════════════════════════════════════════════════════════════════════════════

def render_plotly(ref_mesh, scores, cd_matrix, sim_matrix,
                  mesh_labels, out_dir, obj_name):
    verts  = ref_mesh.vertices.astype(np.float32)
    faces  = ref_mesh.faces
    labels = [l[:10] for l in mesh_labels]
    M      = cd_matrix.shape[0]
    p25,p75 = np.percentile(scores,[25,75])
    off = cd_matrix[~np.eye(M,dtype=bool)]

    fig_mesh = go.Figure(go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        intensity=scores.tolist(), colorscale=cfg.PLOTLY_COLORSCALE,
        colorbar=dict(title=dict(text="Uncertainty",side="right"),
                      thickness=18, tickfont=dict(size=11)),
        lighting=dict(ambient=0.4,diffuse=0.8,specular=0.3,roughness=0.5,fresnel=0.2),
        lightposition=dict(x=1,y=1,z=1),
        hovertemplate="X:%{x:.4f} Y:%{y:.4f} Z:%{z:.4f}"
                      "<br>Uncertainty:%{intensity:.5f}<extra></extra>",
    ))
    fig_mesh.update_layout(
        title=dict(text=f"{obj_name} — Uncertainty (original mesh)",
                   x=0.5, font=dict(size=18,color="white")),
        scene=dict(
            xaxis=dict(backgroundcolor="#111",gridcolor="#333",showbackground=True,tickfont=dict(color="#aaa")),
            yaxis=dict(backgroundcolor="#111",gridcolor="#333",showbackground=True,tickfont=dict(color="#aaa")),
            zaxis=dict(backgroundcolor="#111",gridcolor="#333",showbackground=True,tickfont=dict(color="#aaa")),
            bgcolor="#0d0d0d",camera=dict(eye=dict(x=1.6,y=1.2,z=1.6)),aspectmode="data"),
        paper_bgcolor="#0d0d0d",font=dict(color="white"),
        margin=dict(l=0,r=0,t=55,b=0),height=650)

    fig_cd = go.Figure(go.Heatmap(
        z=cd_matrix, x=labels, y=labels, colorscale="Magma",
        colorbar=dict(title=dict(text="Chamfer",side="right")),
        hovertemplate="A:%{y}<br>B:%{x}<br>CD:%{z:.5f}<extra></extra>"))
    fig_cd.update_layout(
        title=dict(text="Chamfer Distance Matrix",x=0.5,font=dict(size=16,color="white")),
        paper_bgcolor="#0d0d0d",plot_bgcolor="#181818",font=dict(color="white"),
        xaxis=dict(tickangle=-45,tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size
[3/23/2026 5:36 AM] Anish Bishwakarma: =8)),height=530)

    fig_sim = make_subplots(1,2,subplot_titles=["Similarity","MDS Embedding"],
                            horizontal_spacing=0.12)
    fig_sim.add_trace(go.Heatmap(
        z=sim_matrix,x=labels,y=labels,colorscale="Viridis",
        colorbar=dict(title=dict(text="Sim"),x=0.44,thickness=14),
        hovertemplate="A:%{y}<br>B:%{x}<br>Sim:%{z:.4f}<extra></extra>"),1,1)
    try:
        mds = MDS(n_components=2,dissimilarity="precomputed",random_state=42,normalized_stress=False)
        emb = mds.fit_transform(cd_matrix)
    except TypeError:
        mds = MDS(n_components=2,dissimilarity="precomputed",random_state=42)
        emb = mds.fit_transform(cd_matrix)
    fig_sim.add_trace(go.Scatter(
        x=emb[:,0],y=emb[:,1],mode="markers+text",text=labels,
        textposition="top center",textfont=dict(size=7,color="rgba(200,200,200,.8)"),
        marker=dict(size=9,color=list(range(M)),colorscale="Rainbow",
                    showscale=True,colorbar=dict(title=dict(text="Idx"),x=1.02,thickness=14)),
        hovertemplate="%{text}<br>MDS1:%{x:.3f}<br>MDS2:%{y:.3f}<extra></extra>"),1,2)
    fig_sim.update_layout(paper_bgcolor="#0d0d0d",plot_bgcolor="#181818",
                          font=dict(color="white"),height=480,showlegend=False)
    for ann in fig_sim.layout.annotations:
        ann.font.color="white"; ann.font.size=12

    h1 = fig_mesh.to_html(full_html=False, include_plotlyjs="cdn")
    h2 = fig_cd.to_html(full_html=False, include_plotlyjs=False)
    h3 = fig_sim.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>{obj_name} — Uncertainty</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#eee;font-family:'Segoe UI',sans-serif}}
header{{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:20px 36px;border-bottom:1px solid #333}}
header h1{{font-size:1.6rem;font-weight:700;background:linear-gradient(90deg,#e96cff,#55b8ff);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
header p{{color:#aaa;margin-top:4px;font-size:.88rem}}
.tabs{{display:flex;gap:4px;padding:14px 36px 0;border-bottom:2px solid #222}}
.tb{{padding:9px 22px;border:none;border-radius:6px 6px 0 0;cursor:pointer;font-size:.85rem;
     font-weight:600;background:#1e1e2e;color:#aaa;transition:.2s}}
.tb.active{{background:linear-gradient(135deg,#6e4fff,#b44fff);color:#fff}}
.tb:hover:not(.active){{background:#2a2a3e;color:#ddd}}
.tc{{display:none;padding:18px 36px}}.tc.active{{display:block}}
.card{{background:#111;border:1px solid #2a2a2a;border-radius:10px;padding:16px;margin-bottom:16px}}
.card h2{{font-size:.95rem;color:#ccc;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #2a2a2a}}
.sg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:10px}}
.sb{{background:#1a1a2e;border-radius:8px;padding:11px;border:1px solid #2a2a3e}}
.sb .lbl{{font-size:.68rem;color:#888;text-transform:uppercase;letter-spacing:.06em}}
.sb .val{{font-size:1.2rem;font-weight:700;color:#a78bfa;margin-top:2px}}
</style></head><body>
<header>
  <h1>{obj_name}</h1>
  <p>Heatmap on original mesh · {M} generated meshes · metric: {cfg.UNCERTAINTY_METRIC}</p>
</header>
<div class="tabs">
  <button class="tb active" onclick="show('mesh',this)">🔴 3D Heatmap</button>
  <button class="tb"        onclick="show('cd',this)">📐 Chamfer</button>
  <button class="tb"        onclick="show('sim',this)">🔗 Similarity</button>
  <button class="tb"        onclick="show('stats',this)">📊 Stats</button>
</div>
<div id="tab-mesh" class="tc active">
  <div class="card"><h2>Original mesh — drag to rotate · scroll to zoom</h2>{h1}</div></div>
<div id="tab-cd" class="tc"><div class="card"><h2>Chamfer Distance Matrix</h2>{h2}</div></div>
<div id="tab-sim" class="tc"><div class="card"><h2>Similarity & MDS</h2>{h3}</div></div>
<div id="tab-stats" class="tc">
  <div class="card"><h2>Summary Statistics</h2><div class="sg">
    <div
[3/23/2026 5:36 AM] Anish Bishwakarma: <div class
[3/23/2026 5:36 AM] Anish Bishwakarma: ="sg">
    <div cl
[3/23/2026 5:36 AM] Anish Bishwakarma: s="sg">
    <div class="sb"><div class="lbl">Meshes</div><div class="val">{M}</div></div>
    <div class="sb"><div class="lbl">Vertices</div><div class="val">{len(scores):,}</div></div>
    <div class="sb"><div class="lbl">Faces</div><div class="val">{len(ref_mesh.faces):,}</div></div>
    <div class="sb"><div class="lbl">Mean score</div><div class="val">{scores.mean():.5f}</div></div>
    <div class="sb"><div class="lbl">Std</div><div class="val">{scores.std():.5f}</div></div>
    <div class="sb"><div class="lbl">Max</div><div class="val">{scores.max():.5f}</div></div>
    <div class="sb"><div class="lbl">P25</div><div class="val">{p25:.5f}</div></div>
    <div class="sb"><div class="lbl">Median</div><div class="val">{np.median(scores):.5f}</div></div>
    <div class="sb"><div class="lbl">P75</div><div class="val">{p75:.5f}</div></div>
    <div class="sb"><div class="lbl">Mean Chamfer</div><div class="val">{off.mean():.5f}</div></div>
  </div></div></div>
<script>
function show(n,b){{
  document.querySelectorAll('.tc').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tb').forEach(e=>e.classList.remove('active'));
  document.getElementById('tab-'+n).classList.add('active');b.classList.add('active');
}}
</script></body></html>"""
    with open(os.path.join(out_dir,"interactive_heatmap.html"),"w") as f:
        f.write(html)


# ══════════════════════════════════════════════════════════════════════════════
# Stats figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_stats(scores, stats, cd_matrix, mesh_labels, out_dir, obj_name):
    fig = plt.figure(figsize=(17,10), facecolor="#0d0d0d")
    fig.suptitle(f"{obj_name}  —  Uncertainty Statistics",
                 color="white", fontsize=16, fontweight="bold", y=0.99)
    gs = GridSpec(2,3, figure=fig, hspace=0.44, wspace=0.32)
    p25,p75 = np.percentile(scores,[25,75])

    ax1 = fig.add_subplot(gs[0,0], facecolor="#181818")
    ax1.hist(scores, bins=60, color="#e44c7f", edgecolor="none", alpha=0.9)
    ax1.axvline(scores.mean(),     color="#ffd700", lw=1.8, label=f"Mean {scores.mean():.4f}")
    ax1.axvline(np.median(scores), color="#00e5ff", lw=1.8, ls="--",
                label=f"Med {np.median(scores):.4f}")
    ax1.set_title("Distribution", color="white"); _dark(ax1)
    ax1.legend(fontsize=8, labelcolor="white", facecolor="#1e1e1e", edgecolor="#555")

    ax2 = fig.add_subplot(gs[0,1], facecolor="#181818")
    ss  = np.sort(scores); cdf = np.arange(1,len(ss)+1)/len(ss)
    ax2.plot(ss,cdf,color="#a8edea",lw=2); ax2.fill_between(ss,cdf,alpha=0.15,color="#a8edea")
    ax2.axvline(p25,color="#ffd700",lw=1.2,ls=":",label=f"P25 {p25:.4f}")
    ax2.axvline(p75,color="#ff6b6b",lw=1.2,ls=":",label=f"P75 {p75:.4f}")
    ax2.set_title("CDF",color="white"); _dark(ax2)
    ax2.legend(fontsize=8,labelcolor="white",facecolor="#1e1e1e",edgecolor="#555")

    ax3 = fig.add_subplot(gs[0,2], facecolor="#181818")
    md  = [stats["std_displacement"],stats["mean_displacement"],
           stats["range_displacement"],stats["mad_displacement"]]
    pts = ax3.violinplot(md, showmedians=True, showextrema=False)
    for pc in pts["bodies"]: pc.set_facecolor("#6e4fff"); pc.set_alpha(0.7)
    pts["cmedians"].set_color("#ffd700"); pts["cmedians"].set_linewidth(2)
    ax3.set_xticks([1,2,3,4])
    ax3.set_xticklabels(["Std","Mean","Range","MAD"],color="gray",fontsize=9)
    ax3.set_title("Metric Violin",color="white"); _dark(ax3)

    ax4 = fig.add_subplot(gs[1,0], facecolor="#181818")
    pca = PCA(n_components=2); pts2 = pca.fit_transform(np.stack(md,axis=1))
    ax4.scatter(pts2[:,0],pts2[:,1],c=scores_to_rgb_float(scores),s=1,alpha=0.5)
    ax4.set_title(f"PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% var)",color="white")
    _dark(ax4); ax4.set_xlabel("PC1",color="gray"); ax4.set_ylabel("PC2",color="gray")

    ax5 = fig.add_subplot(gs[1,1], facecolor="#181818")
    off = cd_matrix[~np.eye(cd_matrix.shape[0],dtype=bool)]
    ax5.hist(off,bins=40,color="#55b8ff",edgecolor="none",alpha=0.9)
    ax5.set_title("Pairwise
[3/23/2026 5:36 AM] Anish Bishwakarma: Chamfer",color="white"); _dark(ax5)
    ax5.set_xlabel("Chamfer distance",color="gray")

    ax6 = fig.add_subplot(gs[1,2]); ax6.axis("off")
    rows = [["# Meshes",f"{len(mesh_labels)}"],["# Vertices",f"{len(scores):,}"],
            ["Mean",f"{scores.mean():.5f}"],["Std",f"{scores.std():.5f}"],
            ["Min",f"{scores.min():.5f}"],["P25",f"{p25:.5f}"],
            ["Median",f"{np.median(scores):.5f}"],["P75",f"{p75:.5f}"],
            ["Max",f"{scores.max():.5f}"],["IQR",f"{p75-p25:.5f}"],
            ["Chamfer μ",f"{off.mean():.5f}"],["Chamfer σ",f"{off.std():.5f}"]]
    tbl = ax6.table(cellText=rows,colLabels=["Metric","Value"],
                    cellLoc="center",loc="center",bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_facecolor("#1e1e2e" if r>0 else "#333")
        cell.set_text_props(color="white"); cell.set_edgecolor("#444")
    ax6.set_title("Summary",color="white",fontsize=11,pad=10)

    plt.savefig(os.path.join(out_dir,"uncertainty_stats.png"),
                dpi=cfg.FIGURE_DPI,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close()


def plot_similarity_figure(cd_matrix, sim_matrix, mesh_labels, out_dir):
    M = cd_matrix.shape[0]; labels = [l[:9] for l in mesh_labels]
    fig,axes = plt.subplots(1,3,figsize=(20,6),facecolor="#0d0d0d")
    for ax,mat,title,cmap_ in zip(axes[:2],[cd_matrix,sim_matrix],
        ["Chamfer Distance","Similarity"],["magma","viridis"]):
        sns.heatmap(mat,ax=ax,cmap=cmap_,mask=np.eye(M,dtype=bool),
                    xticklabels=labels,yticklabels=labels,
                    linewidths=0.3,linecolor="#222",
                    annot=(M<=20),fmt=".3f",annot_kws={"size":6,"color":"white"},
                    cbar_kws={"shrink":0.8})
        ax.set_title(title,color="white",fontsize=11); ax.tick_params(colors="gray",labelsize=7)
    ax3=axes[2]; ax3.set_facecolor("#181818")
    try:
        mds = MDS(n_components=2,dissimilarity="precomputed",random_state=42,normalized_stress=False)
        emb = mds.fit_transform(cd_matrix)
    except TypeError:
        mds = MDS(n_components=2,dissimilarity="precomputed",random_state=42)
        emb = mds.fit_transform(cd_matrix)
    sc = ax3.scatter(emb[:,0],emb[:,1],c=np.arange(M),cmap="rainbow",
                     s=70,zorder=3,edgecolors="white",linewidths=0.5)
    for i,lbl in enumerate(labels):
        ax3.annotate(lbl,emb[i],fontsize=7,color="gray",
                     xytext=(4,4),textcoords="offset points")
    plt.colorbar(sc,ax=ax3,label="Mesh index").ax.yaxis.set_tick_params(color="white")
    ax3.set_title("MDS Embedding",color="white",fontsize=11); _dark(ax3)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(out_dir,"similarity_matrix.png"),
                dpi=cfg.FIGURE_DPI,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Per-object pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_object(obj_dir: str, obj_name: str) -> Optional[Dict]:
    out_dir = os.path.join(cfg.OUTPUT_ROOT, obj_name)
    os.makedirs(out_dir, exist_ok=True)

    if cfg.SKIP_IF_DONE and os.path.exists(os.path.join(out_dir,"summary.txt")):
        scores = np.load(os.path.join(out_dir,"uncertainty_scores.npy"))
        n = len(glob.glob(os.path.join(obj_dir, cfg.MESH_GLOB)))
        print(f"  [SKIP] '{obj_name}' already done.")
        return _summary(obj_name, scores, n)

    t0 = time.time()
    meshes = load_meshes(obj_dir)
    if len(meshes) < cfg.MIN_MESHES:
        print(f"  [SKIP] '{obj_name}' — only {len(meshes)} mesh(es).")
        return None

    labels = [m.metadata.get("source", f"m{i:03d}") for i,m in enumerate(meshes)]

    # ★ KEY STEP: keep original mesh topology
    ref_mesh, aligned, stacked = build_stacked_on_original(meshes)

    stats   = per_vertex_statistics(stacked)
    scores  = compute_uncertainty(stacked)
    cd_mat  =
[3/23/2026 5:36 AM] Anish Bishwakarma: build_sim
[3/23/2026 5:36 AM] Anish Bishwakarma: ilarity_matrix(sta
[3/23/2026 5:36 AM] Anish Bishwakarma: milarity_matrix(stacked)
    sim_mat = chamfer_to_similarity(cd_mat)

    # Export one colored PLY per input mesh
    ply_paths = export_all_heatmaps(aligned, ref_mesh, scores, out_dir)
    print(f"    → {len(ply_paths)} colored PLY files in colored_ply/")
    render_pyvista(ref_mesh, scores, out_dir, obj_name)
    render_plotly(ref_mesh, scores, cd_mat, sim_mat, labels, out_dir, obj_name)
    plot_stats(scores, stats, cd_mat, labels, out_dir, obj_name)
    plot_similarity_figure(cd_mat, sim_mat, labels, out_dir)

    # Save numerics
    np.save(os.path.join(out_dir,"uncertainty_scores.npy"), scores)
    np.save(os.path.join(out_dir,"chamfer_distance_matrix.npy"), cd_mat)
    for k,v in stats.items():
        if isinstance(v, np.ndarray):
            np.save(os.path.join(out_dir,f"stat_{k}.npy"), v)

    p25,p75 = np.percentile(scores,[25,75])
    off = cd_mat[~np.eye(len(meshes),dtype=bool)]
    with open(os.path.join(out_dir,"summary.txt"),"w") as f:
        f.write(f"Object          : {obj_name}\n")
        f.write(f"Metric          : {cfg.UNCERTAINTY_METRIC}\n")
        f.write(f"# Meshes        : {len(meshes)}\n")
        f.write(f"# Vertices      : {len(scores)}  (original mesh)\n")
        f.write(f"# Faces         : {len(ref_mesh.faces)}\n\n")
        for n,v in [("mean",scores.mean()),("std",scores.std()),("min",scores.min()),
                    ("P25",p25),("median",np.median(scores)),("P75",p75),
                    ("max",scores.max())]:
            f.write(f"  score {n:<8}= {v:.6f}\n")
        f.write(f"\n  chamfer mean   = {off.mean():.6f}\n")

    elapsed = time.time()-t0
    result  = _summary(obj_name, scores, len(meshes),
                       off.mean(), elapsed, len(ref_mesh.faces))
    print(f"  ✓  '{obj_name}'  "
          f"v={len(scores)}  f={len(ref_mesh.faces)}  "
          f"score_mean={scores.mean():.5f}  ({elapsed:.1f}s)")
    return result


def _summary(obj_name, scores, n_meshes,
             chamfer_mean=None, elapsed=None, n_faces=None):
    p25,p75 = np.percentile(scores,[25,75])
    return dict(object=obj_name, n_meshes=n_meshes,
                n_vertices=len(scores), n_faces=int(n_faces or 0),
                score_mean=float(scores.mean()),  score_std=float(scores.std()),
                score_min=float(scores.min()),    score_p25=float(p25),
                score_median=float(np.median(scores)), score_p75=float(p75),
                score_max=float(scores.max()),
                chamfer_mean=float(chamfer_mean) if chamfer_mean else float("nan"),
                elapsed_s=float(elapsed) if elapsed else float("nan"))


# ══════════════════════════════════════════════════════════════════════════════
# Batch report
# ══════════════════════════════════════════════════════════════════════════════

def build_batch_report(summaries: List[Dict]):
    report_dir = os.path.join(cfg.OUTPUT_ROOT,"_batch_report")
    os.makedirs(report_dir, exist_ok=True)

    names    = [s["object"]       for s in summaries]
    means    = [s["score_mean"]   for s in summaries]
    maxs     = [s["score_max"]    for s in summaries]
    stds     = [s["score_std"]    for s in summaries]
    p25s     = [s["score_p25"]    for s in summaries]
    p75s     = [s["score_p75"]    for s in summaries]
    chamfers = [s["chamfer_mean"] for s in summaries]

    # ── Matplotlib comparison ─────────────────────────────────────────────
    fig,axes = plt.subplots(1,3,figsize=(20,7),facecolor="#0d0d0d")
    fig.suptitle("Cross-Object Uncertainty Comparison",
                 color="white",fontsize=17,fontweight="bold",y=0.99)

    order   = np.argsort(means)[::-1]
    names_s = [names[i] for i in order]
    means_s = [means[i] for i in order]

    ax1=axes[0]; ax1.set_facecolor("#181818")
    bars = ax1.barh(names_s, means_s,
                    color=cm.get_cmap("plasma")(Normalize()(means_s)),
                    edgecolor="none", height=0.65)
    ax1.set_xlabel("Mean Uncertainty",color="gray")
    ax1.set_title("Ranking by Mean Score",color="white"); _dark(ax1); ax1.invert_yaxis()
    for bar,v in zip(bars,means_s):
[3/23/2026 5:36 AM] Anish Bishwakarma: ax1.text(v+max(means_s)*0.01, bar.get_y()+bar.get_height()/2,
                 f"{v:.5f}",va="center",ha="left",color="gray",fontsize=8)

    ax2=axes[1]; ax2.set_facecolor("#181818")
    for s in summaries:
        ax2.hlines(s["object"],s["score_p25"],s["score_p75"],
                   color="#a78bfa",linewidth=8,alpha=0.5)
        ax2.scatter(s["score_median"],s["object"],color="#ffd700",s=60,zorder=5)
        ax2.scatter(s["score_max"],   s["object"],color="#ff6b6b",marker="^",s=50,zorder=5)
    ax2.set_xlabel("Score",color="gray")
    ax2.set_title("IQR (bar) · Median (●) · Max (▲)",color="white"); _dark(ax2)

    ax3=axes[2]; ax3.set_facecolor("#181818")
    ax3.bar(names,chamfers,
            color=cm.get_cmap("magma")(Normalize()(chamfers)),edgecolor="none")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names,rotation=40,ha="right",fontsize=9,color="gray")
    ax3.set_title("Mean Chamfer Distance",color="white")
    ax3.set_ylabel("Mean Chamfer",color="gray"); _dark(ax3)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(report_dir,"batch_comparison.png"),
                dpi=cfg.FIGURE_DPI,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close()

    # ── Plotly interactive batch report ───────────────────────────────────
    cpal = px.colors.sample_colorscale("plasma", len(summaries))

    fig1 = go.Figure()
    for trace_name, vals, col in [("Mean",means,"#a78bfa"),
                                   ("Max",maxs,"#ff6b6b"),
                                   ("Std",stds,"#55b8ff")]:
        fig1.add_trace(go.Bar(name=trace_name,x=names,y=vals,marker_color=col,
                              hovertemplate=f"%{{x}}<br>{trace_name}: %{{y:.5f}}<extra></extra>"))
    fig1.update_layout(barmode="group",
        title=dict(text="Per-Object Uncertainty Scores",x=0.5,font=dict(size=17,color="white")),
        paper_bgcolor="#0d0d0d",plot_bgcolor="#181818",font=dict(color="white"),height=450,
        xaxis=dict(tickangle=-30),legend=dict(bgcolor="#1e1e2e",bordercolor="#444"))

    fig2 = go.Figure()
    for i,s in enumerate(summaries):
        sc_path = os.path.join(cfg.OUTPUT_ROOT,s["object"],"uncertainty_scores.npy")
        if os.path.exists(sc_path):
            sc = np.load(sc_path)
            fig2.add_trace(go.Box(y=sc,name=s["object"],boxpoints=False,
                                  marker_color=cpal[i],line_color=cpal[i],
                                  hovertemplate=f"{s['object']}<br>%{{y:.5f}}<extra></extra>"))
    fig2.update_layout(
        title=dict(text="Score Distributions",x=0.5,font=dict(size=17,color="white")),
        paper_bgcolor="#0d0d0d",plot_bgcolor="#181818",font=dict(color="white"),
        height=450,xaxis=dict(tickangle=-30),showlegend=False)

    cats = ["Mean","Std","Max","P75","Chamfer"]
    def nrm(v): mn,mx=min(v),max(v); return [(x-mn)/(mx-mn+1e-9) for x in v]
    fig3 = go.Figure()
    for i,(s,rd) in enumerate(zip(summaries,
        zip(nrm(means),nrm(stds),nrm(maxs),nrm(p75s),nrm(chamfers)))):
        rv = list(rd)+[rd[0]]
        fig3.add_trace(go.Scatterpolar(r=rv,theta=cats+[cats[0]],fill="toself",
                                       name=s["object"],line_color=cpal[i],opacity=0.65))
    fig3.update_layout(
        polar=dict(bgcolor="#181818",
                   radialaxis=dict(visible=True,range=[0,1],color="gray",gridcolor="#333"),
                   angularaxis=dict(color="gray",gridcolor="#333")),
        title=dict(text="Radar: Normalised Metrics",x=0.5,font=dict(size=17,color="white")),
        paper_bgcolor="#0d0d0d",font=dict(color="white"),height=500,
        legend=dict(bgcolor="#1e1e2e",bordercolor="#444",font=dict(size=10)))

    h1=fig1.to_html(full_html=False,include_plotlyjs="cdn")
    h2=fig2.to_html(full_html=False,include_plotlyjs=False)
    h3=fig3.to_html(full_html=False,include_plotlyjs=False)

    cards = ""
    for s in sorted(summaries,key=lambda x:x["score_mean"],reverse=True):
        cards += f"""
<div class="obj-card">
  <a href="../{s['object']}/interactive_heatmap.html" target="_blank">
    <img
[3/23/2026 5:36 AM] Anish Bishwakarma: tive_heatmap.html" target="_blank">
    <img src="../{s['object']}/pyvista_iso.png" alt="{s['object']}"
         onerror="this.style.display='none'">
    <div class="obj-name">{s['object']}</div>
  </a>
  <div class="obj-stats">
    <span>Mean <b>{s['score_mean']:.5f}</b></span>
    <span>Max <b>{s['score_max']:.5f}</b></span>
    <span>{s['n_vertices']:,} verts · {s['n_faces']:,} faces</span>
  </div>
</div>"""

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Batch Uncertainty Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0d0d;color:#eee;font-family:'Segoe UI',sans-serif}}
header{{background:linear-gradient(135deg,#1a1a2e,#16213e);padding:24px 40px;border-bottom:1px solid #333}}
header h1{{font-size:2rem;font-weight:700;background:linear-gradient(90deg,#e96cff,#55b8ff);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
header p{{color:#aaa;margin-top:6px;font-size:.95rem}}
.tabs{{display:flex;gap:4px;padding:16px 40px 0;border-bottom:2px solid #222}}
.tb{{padding:10px 24px;border:none;border-radius:6px 6px 0 0;cursor:pointer;
     font-size:.88rem;font-weight:600;background:#1e1e2e;color:#aaa;transition:.2s}}
.tb.active{{background:linear-gradient(135deg,#6e4fff,#b44fff);color:#fff}}
.tb:hover:not(.active){{background:#2a2a3e;color:#ddd}}
.tc{{display:none;padding:22px 40px}}.tc.active{{display:block}}
.card{{background:#111;border:1px solid #2a2a2a;border-radius:10px;padding:20px;margin-bottom:20px}}
.card h2{{font-size:1rem;color:#ccc;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid #2a2a2a}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}}
.obj-card{{background:#1a1a2e;border-radius:10px;border:1px solid #2a2a3e;overflow:hidden;transition:.2s}}
.obj-card:hover{{border-color:#6e4fff;transform:translateY(-2px)}}
.obj-card a{{text-decoration:none;color:inherit}}
.obj-card img{{width:100%;height:148px;object-fit:cover;display:block;background:#111}}
.obj-name{{font-size:.95rem;font-weight:700;color:#c9b8ff;padding:9px 11px 3px}}
.obj-stats{{padding:3px 11px 10px;font-size:.75rem;color:#888;display:flex;flex-direction:column;gap:2px}}
.obj-stats b{{color:#a78bfa}}
</style></head><body>
<header>
  <h1>Batch Uncertainty Report</h1>
  <p>{len(summaries)} objects · heatmap on original mesh geometry · metric: {cfg.UNCERTAINTY_METRIC}</p>
</header>
<div class="tabs">
  <button class="tb active" onclick="show('gallery',this)">🗂 Objects</button>
  <button class="tb"        onclick="show('bars',this)">📊 Scores</button>
  <button class="tb"        onclick="show('boxes',this)">📦 Distributions</button>
  <button class="tb"        onclick="show('radar',this)">🕸 Radar</button>
</div>
<div id="tab-gallery" class="tc active">
  <div class="card"><h2>Click any card to open its interactive 3D heatmap</h2>
    <div class="gallery">{cards}</div></div></div>
<div id="tab-bars" class="tc">
  <div class="card"><h2>Mean / Max / Std per object</h2>{h1}</div></div>
<div id="tab-boxes" class="tc">
  <div class="card"><h2>Full per-vertex distributions</h2>{h2}</div></div>
<div id="tab-radar" class="tc">
  <div class="card"><h2>Radar — normalised metrics</h2>{h3}</div></div>
<script>
function show(n,b){{
  document.querySelectorAll('.tc').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tb').forEach(e=>e.classList.remove('active'));
  document.getElementById('tab-'+n).classList.add('active');b.classList.add('active');
}}
</script></body></html>"""

    with open(os.path.join(report_dir,"batch_summary.html"),"w") as f:
        f.write(html)

    with open(os.path.join(report_dir,"batch_summary.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader(); w.writerows(summaries)

    print(f"  Saved → _batch_report/batch_summary.html")
    print(f"  Saved → _batch_report/batch_comparison.png")
    print(f"  Saved → _batch_report/batch_summary.csv")


#
[3/23/2026 5:36 AM] Anish Bishwakarma: ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_batch():
    print("="*66)
    print("  Batch Mesh Uncertainty  v3  (heatmap on original mesh)")
    print("="*66)

    obj_dirs = sorted([d for d in Path(cfg.ROOT_DIR).iterdir()
                       if d.is_dir() and not d.name.startswith("_")])
    if not obj_dirs:
        print(f"\n[ERROR] No subfolders in '{cfg.ROOT_DIR}'")
        return

    print(f"\n  {len(obj_dirs)} object folder(s) found in '{cfg.ROOT_DIR}'\n")

    summaries = []
    for i, obj_dir in enumerate(obj_dirs, 1):
        obj_name = obj_dir.name
        n_files  = len(glob.glob(str(obj_dir / cfg.MESH_GLOB)))
        print(f"[{i:02d}/{len(obj_dirs)}]  {obj_name}  ({n_files} meshes)")
        result = process_object(str(obj_dir), obj_name)
        if result:
            summaries.append(result)

    if not summaries:
        print("\n[ERROR] Nothing processed."); return

    print(f"\n[Batch report]")
    build_batch_report(summaries)

    print("\n" + "="*66)
    print(f"  Done!  {len(summaries)} objects.")
    print(f"  Output → {os.path.abspath(cfg.OUTPUT_ROOT)}/")
    print("="*66)
    print("\n  Ranking (highest uncertainty first):")
    for rank, s in enumerate(sorted(summaries,key=lambda x:x["score_mean"],reverse=True),1):
        print(f"  {rank:2d}. {s['object']:<18} "
              f"mean={s['score_mean']:.5f}  max={s['score_max']:.5f}  "
              f"v={s['n_vertices']}  f={s['n_faces']}")


if name == "__main__":
    run_batch()