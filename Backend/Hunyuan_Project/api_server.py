import os
import sys
import uuid
import base64
import subprocess
import shutil
import threading
import re
import json
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import config
import boto3
from botocore.exceptions import ClientError

app = Flask(__name__)
# Enable Cross-Origin requests so the frontend Django platform can communicate with it
CORS(app)

JOB_PROGRESS = {}

# ─── QWEN IMAGE EDIT PIPELINE (lazy loaded) ──────────────────────────────────
_qwen_pipeline = None
_qwen_lock = threading.Lock()

def get_qwen_pipeline():
    """Lazily loads and returns the Qwen Image Edit pipeline. Thread-safe."""
    global _qwen_pipeline
    if _qwen_pipeline is None:
        with _qwen_lock:
            if _qwen_pipeline is None:
                import torch
                from diffusers import QwenImageEditPlusPipeline
                print("[Qwen] Loading Qwen-Image-Edit-2509 pipeline...")
                _qwen_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                    "Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16
                )
                _qwen_pipeline.to('cuda')
                _qwen_pipeline.set_progress_bar_config(disable=None)
                print("[Qwen] Pipeline loaded and ready on CUDA.")
    return _qwen_pipeline

# ─── S3 HELPER ──────────────────────────────────────────────────────────────
_s3_bucket_verified = False
_s3_region_override = None

def _build_s3_client(region_name):
    return boto3.client(
        's3',
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        region_name=region_name
    )

def _log_s3_error(prefix, err):
    err_info = err.response.get('Error', {})
    err_code = err_info.get('Code', 'Unknown')
    err_message = err_info.get('Message', 'Unknown')
    metadata = err.response.get('ResponseMetadata', {})
    status_code = metadata.get('HTTPStatusCode')
    request_id = metadata.get('RequestId')
    host_id = metadata.get('HostId')
    headers = metadata.get('HTTPHeaders', {}) or {}
    bucket_region = headers.get('x-amz-bucket-region')

    print(f"[S3] {prefix}: code={err_code} status={status_code} message={err_message}")
    if bucket_region:
        print(f"[S3] Bucket region from response: {bucket_region}")
    if request_id:
        print(f"[S3] RequestId: {request_id}")
    if host_id:
        print(f"[S3] HostId: {host_id}")

def get_s3_client():
    """Returns a boto3 S3 client and ensures the bucket exists (auto-creates if needed)."""
    global _s3_bucket_verified, _s3_region_override
    region_name = _s3_region_override or config.AWS_REGION_NAME
    client = _build_s3_client(region_name)
    
    if not _s3_bucket_verified:
        try:
            client.head_bucket(Bucket=config.AWS_S3_BUCKET_NAME)
            print(f"[S3] Bucket '{config.AWS_S3_BUCKET_NAME}' exists and accessible.")
        except ClientError as e:
            err_code = str(e.response.get('Error', {}).get('Code', ''))
            headers = e.response.get('ResponseMetadata', {}).get('HTTPHeaders', {}) or {}
            bucket_region = headers.get('x-amz-bucket-region')

            if bucket_region and bucket_region != region_name:
                print(
                    f"[S3] Bucket region mismatch. Requested '{region_name}', bucket in '{bucket_region}'. Retrying..."
                )
                _s3_region_override = bucket_region
                client = _build_s3_client(bucket_region)
                try:
                    client.head_bucket(Bucket=config.AWS_S3_BUCKET_NAME)
                    print(f"[S3] Bucket '{config.AWS_S3_BUCKET_NAME}' exists and accessible.")
                except ClientError as retry_err:
                    _log_s3_error("Bucket access error after region retry", retry_err)
                    raise
            elif err_code in ('404', 'NoSuchBucket', 'NotFound'):
                print(f"[S3] Bucket '{config.AWS_S3_BUCKET_NAME}' not found. Creating...")
                try:
                    if config.AWS_REGION_NAME == 'us-east-1':
                        client.create_bucket(Bucket=config.AWS_S3_BUCKET_NAME)
                    else:
                        client.create_bucket(
                            Bucket=config.AWS_S3_BUCKET_NAME,
                            CreateBucketConfiguration={'LocationConstraint': config.AWS_REGION_NAME}
                        )
                    print(f"[S3] Bucket '{config.AWS_S3_BUCKET_NAME}' created successfully!")
                except Exception as create_err:
                    print(f"[S3] Failed to create bucket: {create_err}")
                    raise
            else:
                _log_s3_error("Bucket access error", e)
                raise
        _s3_bucket_verified = True
    
    return client

# Ensure required directories exist
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)

# ─── STATIC ROUTES ──────────────────────────────────────────────────────────
@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serves the generated model files back to the frontend."""
    return send_from_directory(config.OUTPUT_DIR, filename)

# ─── API ROUTES ─────────────────────────────────────────────────────────────

def run_job_and_webhook(cmd, session_id, webhook_url, expected_paths, job_type="3d"):
    """
    Background worker that runs the heavy GPU pipeline.
    Parses live stdout to deduce rendering progress actively into JOB_PROGRESS array.
    """
    JOB_PROGRESS[session_id] = {"progress": 10, "message": "Booting Geometry Engines..."}
    try:
        print(f"\n[{session_id}] Worker Thread Started. Triggering Pipeline: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            
            # Simple live log analyzer
            line_lower = line.lower()
            if "pull" in line_lower and "downloading" in line_lower:
                JOB_PROGRESS[session_id] = {"progress": 15, "message": "Pulling Docker Layers..."}
            elif "removing background" in line_lower or "rembg" in line_lower:
                JOB_PROGRESS[session_id] = {"progress": 25, "message": "Segmenting Input Image..."}
            elif "[shape] seed" in line_lower:
                JOB_PROGRESS[session_id] = {"progress": 40, "message": "Synthesizing 3D Geometry Framework..."}
            elif "[textured] seed" in line_lower:
                JOB_PROGRESS[session_id] = {"progress": 70, "message": "Applying Paint / Albedo PBR..."}
            elif "batch complete" in line_lower or "done!" in line_lower:
                JOB_PROGRESS[session_id] = {"progress": 90, "message": "Exporting Mesh Components..."}
                
        proc.wait()
        
        if proc.returncode != 0:
            print(f"[{session_id}] ERROR: Pipeline script failed.")
            JOB_PROGRESS[session_id] = {"progress": 0, "message": "Rendering crashed! Check Backend logs."}
            return
            
        final_glb_path = None
        for path in expected_paths:
            if os.path.exists(path):
                final_glb_path = path
                break
                
        if not final_glb_path:
            print(f"[{session_id}] ERROR: Could not find generated GLB in expected paths.")
            JOB_PROGRESS[session_id] = {"progress": 0, "message": "Missing Output Asset Error."}
            return
            
        use_s3 = not getattr(config, "BYPASS_S3_UPLOAD", False)
        if use_s3:
            print(f"[{session_id}] SUCCESS: Generation complete. Transmitting to AWS S3...")
            JOB_PROGRESS[session_id] = {"progress": 95, "message": "Uploading 3D Asset to Cloud Storage..."}
            
            s3_client = get_s3_client()
            
            # Uniquely identify the mesh
            s3_key = f"3d_asset_{session_id}.glb"
            
            print(f"[{session_id}] Pushing directly to AWS bucket '{config.AWS_S3_BUCKET_NAME}' as '{s3_key}'...")
            s3_client.upload_file(
                final_glb_path,
                config.AWS_S3_BUCKET_NAME,
                s3_key,
                ExtraArgs={'ContentType': 'model/gltf-binary'}
            )
            
            payload = {
                "s3_key": s3_key,
                "job_type": job_type,
                "session_id": session_id
            }
            
            print(f"[{session_id}] S3 Upload Successful. Triggering Frontend Django Webhook Ping...")
        else:
            print(f"[{session_id}] SUCCESS: Generation complete. Bypassing S3 upload.")
            JOB_PROGRESS[session_id] = {"progress": 95, "message": "Finalizing 3D Asset..."}
            
            rel_path = os.path.relpath(final_glb_path, config.OUTPUT_DIR).replace(os.sep, '/')
            payload = {
                "asset_path": rel_path,
                "job_type": job_type,
                "session_id": session_id
            }
            
            print(f"[{session_id}] Sending asset path to frontend webhook: {rel_path}")
        
        import urllib.request
        payload = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(webhook_url, data=payload)
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-Session-Id', session_id)
        req.add_header('X-Job-Type', job_type)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = response.read().decode('utf-8')
                print(f"[{session_id}] WEBHOOK DELIVERED: {res_data}")
                JOB_PROGRESS[session_id] = {"progress": 100, "message": "Complete!"}
        except urllib.error.URLError as e:
            print(f"[{session_id}] WEBHOOK DELIVERY FAILED: {e}")
            if use_s3:
                JOB_PROGRESS[session_id] = {"progress": 100, "message": f"Saved to S3, but Frontend Webhook failed: {e}"}
            else:
                JOB_PROGRESS[session_id] = {"progress": 100, "message": f"Asset ready, but Frontend Webhook failed: {e}"}
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        JOB_PROGRESS[session_id] = {"progress": 0, "message": f"Critical Error: {str(e)[:100]}"}
        print(f"[{session_id}] BACKGROUND THREAD CRASH: {e}")

@app.route('/api/generate', methods=['POST'])
def generate_3d():
    """
    Main API endpoint:
    1. Receives Base64 image and webhook_url
    2. Spawns pipeline thread
    3. Exits immediately to prevent timeouts
    """
    try:
        data = request.json
        image_data = data.get('image')
        session_id = data.get('session_id')
        webhook_url = data.get('webhook_url')
        
        if not all([image_data, session_id, webhook_url]):
            return jsonify({'success': False, 'error': 'Missing image, session_id, or webhook_url'}), 400
            
        
        if ';base64,' in image_data:
            _, img_str = image_data.split(';base64,')
        else:
            img_str = image_data
            
        img_bytes = base64.b64decode(img_str)
        stem = f"asset_{session_id}"
        
        input_filename = f"{stem}.jpg"
        input_path = os.path.join(config.OUTPUT_DIR, input_filename)
        
        with open(input_path, "wb") as f:
            f.write(img_bytes)
            
        cmd = [
            "bash", config.RUNNER_SCRIPT,
            input_path,
            config.DEFAULT_MODE,
            "--num-outputs", str(config.NUM_OUTPUTS),
            "-o", config.OUTPUT_DIR,
            "-d", config.DOCKER_IMAGE,
            "--cache-dir", config.CACHE_DIR
        ]
        
        # Expected outputs
        expected = [
            os.path.join(config.OUTPUT_DIR, 'outputs', stem, f"{stem}_1", f"{stem}_texture.glb"),
            os.path.join(config.OUTPUT_DIR, 'outputs', stem, f"{stem}_1", f"{stem}_shape.glb")
        ]
        
        # Fire and forget
        thread = threading.Thread(target=run_job_and_webhook, args=(cmd, session_id, webhook_url, expected, "3d"))
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '3D Asset generation queued securely',
            'session_id': session_id
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-heatmap', methods=['POST'])
def generate_heatmap():
    """
    1. Locates the existing session's jpg
    2. Spawns batch Hunyuan3D thread
    3. Exits immediately
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        webhook_url = data.get('webhook_url')
        
        if not all([session_id, webhook_url]):
            return jsonify({'success': False, 'error': 'Missing session_id or webhook_url'}), 400
            
        stem = f"asset_{session_id}"
        input_filename = f"{stem}.jpg"
        input_path = os.path.join(config.OUTPUT_DIR, input_filename)
        
        if not os.path.exists(input_path):
            return jsonify({'success': False, 'error': 'Original image not found for this session.'}), 404
            
        HEATMAP_FACES = 3
        
        cmd_generate = [
            "bash", config.RUNNER_SCRIPT,
            input_path,
            config.DEFAULT_MODE,
            "--num-outputs", str(HEATMAP_FACES),
            "-o", config.OUTPUT_DIR,
            "-d", config.DOCKER_IMAGE,
            "--cache-dir", config.CACHE_DIR
        ]
        
        input_dir = os.path.join(config.OUTPUT_DIR, "outputs", stem)
        final_heatmap_path = os.path.join(config.OUTPUT_DIR, f"heatmap_{session_id}.glb")
        
        cmd_heatmap = [
            "python", "compute_glb_heatmap.py",
            input_dir,
            final_heatmap_path
        ]
        
        # Chain execution in a threaded wrapper
        def run_heatmap_job():
            JOB_PROGRESS[session_id] = {"progress": 10, "message": "Initiating batch multi-mesh variation pipeline..."}
            print(f"\n[{session_id}] Thread: Heatmap batch trigger")
            
            proc1 = subprocess.Popen(cmd_generate, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc1.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                # Parse iterations from standard output dynamically
                match = re.search(r'Iteration (\d+)/', line)
                if match:
                    itr = int(match.group(1))
                    if itr > 0:
                        JOB_PROGRESS[session_id] = {"progress": 10 + (itr*15), "message": f"Rendering Structural Variance Iteration {itr} of {HEATMAP_FACES}..."}
                        
            proc1.wait()
            
            if proc1.returncode == 0:
                import glob as glob_mod
                import urllib.request
                
                use_s3 = not getattr(config, "BYPASS_S3_UPLOAD", False)
                s3_client = get_s3_client() if use_s3 else None
                iteration_data = []
                
                # ── Step A: Upload individual iteration GLBs ──
                if use_s3:
                    JOB_PROGRESS[session_id] = {"progress": 65, "message": "Uploading individual iteration meshes..."}
                    print(f"[{session_id}] Thread: Uploading individual iteration GLBs...")
                    
                    iter_glbs = sorted(glob_mod.glob(os.path.join(input_dir, f"{stem}_*", f"{stem}_texture.glb")))
                    if not iter_glbs:
                        iter_glbs = sorted(glob_mod.glob(os.path.join(input_dir, f"{stem}_*", f"{stem}_shape.glb")))
                    
                    for glb_path in iter_glbs:
                        folder = os.path.basename(os.path.dirname(glb_path))
                        parts = folder.rsplit('_', 1)
                        if len(parts) == 2 and parts[1].isdigit():
                            iter_num = int(parts[1])
                            iter_s3_key = f"iter_{session_id}_{iter_num}.glb"
                            try:
                                s3_client.upload_file(glb_path, config.AWS_S3_BUCKET_NAME, iter_s3_key, ExtraArgs={'ContentType': 'model/gltf-binary'})
                                iteration_data.append({"number": iter_num, "glb_key": iter_s3_key})
                                print(f"[{session_id}]   Uploaded iteration {iter_num} GLB → {iter_s3_key}")
                            except Exception as e:
                                print(f"[{session_id}]   Failed to upload iteration {iter_num}: {e}")
                else:
                    print(f"[{session_id}] Thread: Skipping iteration uploads (S3 bypass enabled).")
                
                # ── Step B: Compute per-iteration heatmaps ──
                JOB_PROGRESS[session_id] = {"progress": 75, "message": "Computing per-iteration topology maps..."}
                print(f"[{session_id}] Thread: Computing per-iteration heatmaps...")
                
                try:
                    import trimesh
                    import numpy as np
                    from scipy.spatial import cKDTree
                    import matplotlib.cm as cm
                    from matplotlib.colors import Normalize
                    
                    obj_paths = sorted(glob_mod.glob(os.path.join(input_dir, f"{stem}_*", "textured_mesh.obj")))
                    if len(obj_paths) < 2:
                        obj_paths = sorted(glob_mod.glob(os.path.join(input_dir, f"{stem}_*", "mesh.obj")))
                    
                    iter_meshes = []
                    for p in obj_paths:
                        folder_name = os.path.basename(os.path.dirname(p))
                        p_parts = folder_name.rsplit('_', 1)
                        if len(p_parts) == 2 and p_parts[1].isdigit():
                            try:
                                m = trimesh.load(p, force="mesh", process=True)
                                if isinstance(m, trimesh.Scene) and m.geometry:
                                    m = trimesh.util.concatenate(list(m.geometry.values()))
                                if m and hasattr(m, 'vertices') and len(m.vertices) > 0:
                                    iter_meshes.append({'number': int(p_parts[1]), 'mesh': m})
                            except Exception as e:
                                print(f"[{session_id}]   Failed loading mesh {p}: {e}")
                    
                    if len(iter_meshes) >= 2:
                        ref_mesh = iter_meshes[-1]['mesh']
                        tree = cKDTree(ref_mesh.vertices)
                        
                        for it_m in iter_meshes:
                            try:
                                distances, _ = tree.query(it_m['mesh'].vertices)
                                vmax = np.percentile(distances, 98) if len(distances) > 0 else 0.1
                                norm = Normalize(vmin=0, vmax=max(vmax, 0.001))
                                cmap_fn = cm.get_cmap("plasma")
                                rgba = (cmap_fn(norm(distances)) * 255).astype(np.uint8)
                                it_m['mesh'].visual = trimesh.visual.ColorVisuals(mesh=it_m['mesh'], vertex_colors=rgba)
                                
                                hm_path = os.path.join(config.OUTPUT_DIR, f"itermap_{session_id}_{it_m['number']}.glb")
                                it_m['mesh'].export(hm_path)
                                
                                if use_s3:
                                    hm_s3_key = f"itermap_{session_id}_{it_m['number']}.glb"
                                    s3_client.upload_file(hm_path, config.AWS_S3_BUCKET_NAME, hm_s3_key, ExtraArgs={'ContentType': 'model/gltf-binary'})
                                    
                                    for d in iteration_data:
                                        if d['number'] == it_m['number']:
                                            d['heatmap_key'] = hm_s3_key
                                            break
                                    print(f"[{session_id}]   Computed + uploaded heatmap for iteration {it_m['number']}")
                                else:
                                    print(f"[{session_id}]   Computed heatmap for iteration {it_m['number']} (upload skipped)")
                            except Exception as e:
                                print(f"[{session_id}]   Failed heatmap for iteration {it_m['number']}: {e}")
                    else:
                        print(f"[{session_id}]   Not enough meshes for per-iteration heatmaps ({len(iter_meshes)} found)")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[{session_id}] Per-iteration heatmap computation failed: {e}")
                
                # ── Step C: Compute combined heatmap (existing logic) ──
                JOB_PROGRESS[session_id] = {"progress": 85, "message": "Compiling combined uncertainty heatmap..."}
                print(f"[{session_id}] Thread: Compiling combined heatmap...")
                
                proc2 = subprocess.Popen(cmd_heatmap, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc2.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                proc2.wait()
                
                if os.path.exists(final_heatmap_path):
                    if use_s3:
                        JOB_PROGRESS[session_id] = {"progress": 95, "message": "Pushing all assets to Cloud Storage..."}
                        
                        s3_key = f"heatmap_asset_{session_id}.glb"
                        print(f"[{session_id}] Uploading combined heatmap to AWS '{config.AWS_S3_BUCKET_NAME}'...")
                        s3_client.upload_file(
                            final_heatmap_path,
                            config.AWS_S3_BUCKET_NAME,
                            s3_key,
                            ExtraArgs={'ContentType': 'model/gltf-binary'}
                        )
                        
                        # Send enriched webhook with iteration data
                        payload = json.dumps({
                            "s3_key": s3_key,
                            "job_type": "heatmap",
                            "session_id": session_id,
                            "iterations": iteration_data
                        }).encode('utf-8')
                        
                        print(f"[{session_id}] Upload complete. Sending enriched webhook with {len(iteration_data)} iterations...")
                    else:
                        JOB_PROGRESS[session_id] = {"progress": 95, "message": "Finalizing heatmap asset..."}
                        rel_path = os.path.relpath(final_heatmap_path, config.OUTPUT_DIR).replace(os.sep, '/')
                        payload = json.dumps({
                            "asset_path": rel_path,
                            "job_type": "heatmap",
                            "session_id": session_id
                        }).encode('utf-8')
                        
                        print(f"[{session_id}] Heatmap ready. Sending direct asset path: {rel_path}")
                    req = urllib.request.Request(webhook_url, data=payload)
                    req.add_header('Content-Type', 'application/json')
                    req.add_header('X-Session-Id', session_id)
                    req.add_header('X-Job-Type', 'heatmap')
                    
                    try:
                        with urllib.request.urlopen(req, timeout=30) as r:
                            print(f"[{session_id}] WEBHOOK DELIVERED.")
                            JOB_PROGRESS[session_id] = {"progress": 100, "message": "Complete!"}
                    except Exception as e:
                        print(f"[{session_id}] Failed webhook: {e}")
                        JOB_PROGRESS[session_id] = {"progress": 100, "message": f"Saved to S3, but Frontend Webhook failed: {str(e)[:80]}"}
                else:
                    print(f"[{session_id}] Heatmap compiler failed.")
                    JOB_PROGRESS[session_id] = {"progress": 0, "message": "Heatmap compilation failed."}
            else:
                JOB_PROGRESS[session_id] = {"progress": 0, "message": "GPU Rendering Thread Crashed."}
                    
        threading.Thread(target=run_heatmap_job).start()
        
        return jsonify({
            'success': True,
            'message': 'Uncertainty Heatmap processing queued.',
            'session_id': session_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status/<session_id>', methods=['GET'])
def get_job_status(session_id):
    """
    Called implicitly by Django frontend to pipe live rendering percentages 
    seamlessly back down the pipe proxy into Javascript UI elements.
    """
    if session_id in JOB_PROGRESS:
        return jsonify({"success": True, "status": JOB_PROGRESS[session_id]})
    return jsonify({"success": False, "error": "Session Not Tracked (Might have instantly cached)"})



# ─── IMAGE EDITING (QWEN) ────────────────────────────────────────────────────

EDIT_RESULTS = {}  # Stores edit job progress and results by edit_id

@app.route('/api/edit-image', methods=['POST'])
def edit_image():
    """
    Accepts image + prompt, spawns a background thread for Qwen editing,
    returns immediately with an edit_id for polling.
    """
    try:
        data = request.json
        image_data = data.get('image')
        prompt = data.get('prompt', '')
        seed = data.get('seed', 0)

        if not image_data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        if not prompt.strip():
            return jsonify({'success': False, 'error': 'No edit prompt provided'}), 400

        edit_id = str(uuid.uuid4())[:8]
        EDIT_RESULTS[edit_id] = {"status": "processing", "progress": 5, "message": "Queued for editing..."}

        print(f"[Qwen] Edit request queued as {edit_id}. Prompt: '{prompt}'")

        def run_edit():
            try:
                import torch

                EDIT_RESULTS[edit_id] = {"status": "processing", "progress": 10, "message": "Decoding input image..."}

                # Decode the incoming base64 image
                if ';base64,' in image_data:
                    _, img_str = image_data.split(';base64,')
                else:
                    img_str = image_data

                img_bytes = base64.b64decode(img_str)
                input_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

                print(f"[Qwen][{edit_id}] Image decoded. Size: {input_image.size}")
                EDIT_RESULTS[edit_id] = {"status": "processing", "progress": 20, "message": "Loading Qwen AI model..."}

                pipeline = get_qwen_pipeline()

                EDIT_RESULTS[edit_id] = {"status": "processing", "progress": 30, "message": "Running AI image edit (this takes ~1-2 min)..."}

                num_steps = 40

                # Callback to report per-step diffusion progress (30% → 90%)
                def step_callback(pipe, step_index, timestep, callback_kwargs):
                    progress = 30 + int((step_index + 1) / num_steps * 60)  # Maps 0..39 → 30..90
                    EDIT_RESULTS[edit_id] = {
                        "status": "processing",
                        "progress": progress,
                        "message": f"Diffusion step {step_index + 1}/{num_steps}..."
                    }
                    return callback_kwargs

                inputs = {
                    "image": [input_image],
                    "prompt": prompt,
                    "generator": torch.manual_seed(int(seed)),
                    "true_cfg_scale": 4.0,
                    "negative_prompt": " ",
                    "num_inference_steps": num_steps,
                    "guidance_scale": 1.0,
                    "num_images_per_prompt": 1,
                    "callback_on_step_end": step_callback,
                }

                with torch.inference_mode():
                    output = pipeline(**inputs)
                    output_image = output.images[0]

                EDIT_RESULTS[edit_id] = {"status": "processing", "progress": 90, "message": "Encoding result..."}

                buffer = io.BytesIO()
                output_image.save(buffer, format='PNG')
                buffer.seek(0)
                encoded = base64.b64encode(buffer.read()).decode('utf-8')
                result_data_url = f"data:image/png;base64,{encoded}"

                print(f"[Qwen][{edit_id}] Edit complete. Output size: {output_image.size}")

                EDIT_RESULTS[edit_id] = {
                    "status": "completed",
                    "progress": 100,
                    "message": "Edit complete!",
                    "edited_image": result_data_url
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Qwen][{edit_id}] Edit failed: {e}")
                EDIT_RESULTS[edit_id] = {"status": "error", "progress": 0, "message": str(e)[:200]}

        threading.Thread(target=run_edit, daemon=True).start()

        return jsonify({
            'success': True,
            'edit_id': edit_id,
            'message': 'Image editing started'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _schedule_edit_cleanup(edit_id, delay=60):
    """Schedules cleanup of a completed/error edit result after a delay.
    This prevents the result from being lost if the frontend polls multiple times."""
    def _cleanup():
        EDIT_RESULTS.pop(edit_id, None)
    timer = threading.Timer(delay, _cleanup)
    timer.daemon = True
    timer.start()

@app.route('/api/edit-status/<edit_id>', methods=['GET'])
def get_edit_status(edit_id):
    """
    Polling endpoint for image edit progress. Returns the edited image
    as base64 when complete.
    """
    if edit_id not in EDIT_RESULTS:
        return jsonify({"success": False, "error": "Edit job not found"}), 404

    result = EDIT_RESULTS[edit_id]

    if result["status"] == "completed":
        edited_image = result.get("edited_image", "")
        # Schedule cleanup after 60s so retries/race conditions still get the result
        _schedule_edit_cleanup(edit_id, delay=60)
        return jsonify({
            "success": True,
            "status": "completed",
            "progress": 100,
            "message": "Edit complete!",
            "edited_image": edited_image
        })
    elif result["status"] == "error":
        msg = result.get("message", "Unknown error")
        _schedule_edit_cleanup(edit_id, delay=30)
        return jsonify({"success": False, "status": "error", "progress": 0, "message": msg})
    else:
        return jsonify({
            "success": True,
            "status": "processing",
            "progress": result.get("progress", 0),
            "message": result.get("message", "Processing...")
        })


# ─── RUN SERVER & EXPOSE TUNNEL ──────────────────────────────────────────────

def ensure_cloudflared_binary():
    """
    Checks if `cloudflared` exists in the system PATH.
    If not, downloads the standard linux-amd64 executable locally into CACHE_DIR without needing sudo.
    """
    if shutil.which("cloudflared") is not None:
        return "cloudflared"
        
    local_binary = os.path.join(config.CACHE_DIR, "cloudflared")
    if os.path.exists(local_binary):
        return local_binary
        
    import urllib.request
    print("\n[Setup] 'cloudflared' not found globally. Auto-downloading portable binary to cache...")
    try:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        urllib.request.urlretrieve(url, local_binary)
        os.chmod(local_binary, 0o755) # Make it executable
        print("[Setup] Download complete and ready to tunnel!\n")
        return local_binary
    except Exception as e:
        print(f"[Setup Error] Failed to download cloudflared: {e}")
        return "cloudflared"  # Fallback to trigger the usual error

def start_tunnel():
    """Attempts to span the backend to the Internet via Cloudflare Tunnel"""
    try:
        print("====== TUNNEL ======")
        print(f"Starting Cloudflared Tunnel to port {config.PORT}...")
        
        binary_path = ensure_cloudflared_binary()
        
        # Use Popen to intercept the tunneling process output
        proc = subprocess.Popen(
            [binary_path, "tunnel", "--url", f"http://localhost:{config.PORT}"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True
        )
        
        def monitor_tunnel():
            url_found = False
            for line in proc.stdout:
                if not url_found:
                    # Cloudflare prints URLs in the format "https://[words].trycloudflare.com"
                    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                    if match:
                        tunnel_url = match.group(0)
                        print("\n" + "★" * 60)
                        print(f"  SUCCESS! YOUR PUBLIC API URL IS: ")
                        print(f"  -->  {tunnel_url}  <--")
                        print("  Copy this URL and place it in your frontend's config.py!")
                        print("★" * 60 + "\n")
                        url_found = True
        
        # Start a daemon thread to read the tunnel logs asynchronously
        t = threading.Thread(target=monitor_tunnel, daemon=True)
        t.start()
        
    except FileNotFoundError:
        print("⚠️ 'cloudflared' not found. We could not auto-download it for tunneling.")
    except Exception as e:
        print(f"Tunnel Initialization Error: {e}")

if __name__ == '__main__':
    if config.USE_CLOUDFLARE_TUNNEL:
        start_tunnel()
    
    # Run server locally (cloudflared routes traffic into this port)
    app.run(host=config.HOST, port=config.PORT, debug=False)
