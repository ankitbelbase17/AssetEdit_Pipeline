import os
import glob
import sys
import numpy as np
import trimesh
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def process_artifact(artifact_dir, output_dir):
    artifact_name = os.path.basename(artifact_dir).lower().replace(' ', '_')
    
    # Collect all iterations
    # e.g. BG_removed_ouptuts_hunyuan/kanha/Kanha_1/textured_mesh.obj
    iter_folders = glob.glob(os.path.join(artifact_dir, "*_*"))
    iterations = []
    
    for folder in iter_folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            obj_path = os.path.join(folder, "textured_mesh.obj")
            if os.path.exists(obj_path):
                iterations.append({
                    'number': int(parts[1]),
                    'path': obj_path
                })
                
    if not iterations:
        print(f"No valid iterations found in {artifact_dir}")
        return
        
    iterations.sort(key=lambda x: x['number'])
    
    # Target reference mesh is the final iteration
    ref_iter = iterations[-1]
    ref_mesh = None
    try:
        ref_mesh = trimesh.load(ref_iter['path'], force="mesh", process=True)
        if hasattr(ref_mesh, 'geometry') and ref_mesh.geometry:
            ref_mesh = trimesh.util.concatenate(list(ref_mesh.geometry.values()))
    except Exception as e:
        print(f"Failed to load reference mesh {ref_iter['path']}: {e}")
        return
        
    if ref_mesh is None or not hasattr(ref_mesh, 'vertices') or len(ref_mesh.vertices) == 0:
        print(f"Reference mesh is empty: {ref_iter['path']}")
        return

    # Prepare output directory
    art_out_dir = os.path.join(output_dir, artifact_name)
    os.makedirs(art_out_dir, exist_ok=True)
    
    print(f"Processing {artifact_name} ({len(iterations)} iterations). Reference: Iteration {ref_iter['number']}")

    # Create proximity query object heavily speeding up closest_point mapping
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_mesh.vertices)
    
    for i, it in enumerate(iterations):
        out_glb = os.path.join(art_out_dir, f"iter_{it['number']}.glb")
        
        # Skip if already exists to save time during multiple script runs
        if os.path.exists(out_glb):
            print(f"  Skipping Iter {it['number']} (already generated)")
            continue
            
        print(f"  Processing Iter {it['number']}...")
        try:
            cur_mesh = trimesh.load(it['path'], force="mesh", process=True)
            if hasattr(cur_mesh, 'geometry') and cur_mesh.geometry:
                cur_mesh = trimesh.util.concatenate(list(cur_mesh.geometry.values()))
                
            if cur_mesh is None or not hasattr(cur_mesh, 'vertices') or len(cur_mesh.vertices) == 0:
                print(f"    Empty mesh for iter {it['number']}")
                continue
                
            # Find distances from each vertex of the current mesh to the reference mesh
            distances, _ = tree.query(cur_mesh.vertices)
            
            # Map distance values to Plasma colormap
            # We cap maximum deviation visually at a threshold to keep map highly responsive
            vmax = np.percentile(distances, 98) if len(distances) > 0 else 0.1
            norm = Normalize(vmin=0, vmax=max(vmax, 0.001))
            cmap = cm.get_cmap("plasma")
            rgba = (cmap(norm(distances)) * 255).astype(np.uint8)
            
            # Set vertex colors natively to map heavily cleanly on GLTF
            cur_mesh.visual = trimesh.visual.ColorVisuals(mesh=cur_mesh, vertex_colors=rgba)
            
            # Export tightly to binary format natively understood by Model-Viewer
            cur_mesh.export(out_glb)
            
        except Exception as e:
            print(f"    Failed iter {it['number']}: {e}")


def main():
    base_dir = r"c:\Users\acer\OneDrive\Desktop\final_year\BG_removed_ouptuts_hunyuan"
    output_dir = r"c:\Users\acer\OneDrive\Desktop\final_year\Project-Display\static\comparison"
    
    if not os.path.exists(base_dir):
        print(f"Could not find {base_dir}")
        sys.exit(1)
        
    # Discover all artifacts
    artifacts = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for art_dir in artifacts:
        process_artifact(art_dir, output_dir)
        
    print("\nAll Heatmaps Generated Successfully into the Frontend Comparison Engine!")


if __name__ == "__main__":
    main()
