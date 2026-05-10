import sys
import os
import glob
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def main(input_dir, output_glb):
    # Find all generated iterations via their pure OBJ representations natively to bypass rigid trimesh GLB header mismatches
    paths = glob.glob(os.path.join(input_dir, "*", "textured_mesh.obj"))
    if len(paths) < 2:
        paths = glob.glob(os.path.join(input_dir, "*", "mesh.obj"))
    
    if len(paths) < 2:
        print("Not enough meshes found for heatmap generation (need at least 2).")
        sys.exit(1)
        
    print(f"Loaded {len(paths)} meshes for uncertainty calculation...")
    
    meshes = []
    for p in paths:
        try:
            m = trimesh.load(p, force="mesh", process=True)
            # Handle if trimesh loads a Scene instead of a single Mesh
            if isinstance(m, trimesh.Scene):
                # Concatenate all geometries into one mesh
                if m.geometry:
                    m = trimesh.util.concatenate(list(m.geometry.values()))
                else:
                    m = None
                    
            if m and hasattr(m, 'vertices') and len(m.vertices) > 0:
                meshes.append(m)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            
    if len(meshes) < 2:
        print("Not enough valid meshes extracted for heatmap.")
        sys.exit(1)
        
    ref = meshes[0]
    V = len(ref.vertices)
    M = len(meshes)
    stacked = np.zeros((M, V, 3), dtype=np.float32)
    stacked[0] = ref.vertices
    
    # Map vertices from reference mesh to nearest points on all other generated iterations
    # (Since Hunyuan3D iterations share the same world-space camera/pose, we bypass ICP logic here for speed)
    for i, mesh in enumerate(meshes[1:], 1):
        # Sample heavily to get good nearest neighbor points
        sample_size = max(V * 8, 20000)
        pts, _ = trimesh.sample.sample_surface(mesh, sample_size)
        _, idx = cKDTree(pts).query(ref.vertices, workers=-1)
        stacked[i] = pts[idx]
        
    # Calculate variation per vertex (Standard Deviation)
    dist = np.linalg.norm(stacked - stacked.mean(axis=0)[np.newaxis], axis=2)
    scores = dist.std(axis=0).astype(np.float32)
    
    # Map to Plasma colormap
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = cm.get_cmap("plasma")
    rgba = (cmap(norm(scores)) * 255).astype(np.uint8)
    
    # Apply colours directly to the vertices
    ref.visual = trimesh.visual.ColorVisuals(mesh=ref, vertex_colors=rgba)
    
    # Export explicitly as GLB
    ref.export(output_glb)
    print(f"Heatmap written successfully to {output_glb}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compute_glb_heatmap.py <input_dir> <output.glb>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
