import os
import sys
import uuid
import base64
import subprocess
import shutil
import threading
import re
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import config
import boto3

app = Flask(__name__)
# Enable Cross-Origin requests so the frontend Django platform can communicate with it
CORS(app)

JOB_PROGRESS = {}

# ─── S3 HELPER ──────────────────────────────────────────────────────────────
_s3_bucket_verified = False

def get_s3_client():
    """Returns a boto3 S3 client and ensures the bucket exists (auto-creates if needed)."""
    global _s3_bucket_verified
    client = boto3.client(
        's3',
        aws_access_key_id=config.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        region_name=config.AWS_REGION_NAME
    )
    
    if not _s3_bucket_verified:
        try:
            client.head_bucket(Bucket=config.AWS_S3_BUCKET_NAME)
            print(f"[S3] Bucket '{config.AWS_S3_BUCKET_NAME}' exists and accessible.")
        except client.exceptions.ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
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
                print(f"[S3] Bucket access error (code {error_code}): {e}")
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
        
        import urllib.request
        payload = json.dumps({
            "s3_key": s3_key,
            "job_type": job_type,
            "session_id": session_id
        }).encode('utf-8')
        
        print(f"[{session_id}] S3 Upload Successful. Triggering Frontend Django Webhook Ping...")
        req = urllib.request.Request(webhook_url, data=payload)
        req.add_header('Content-Type', 'application/json')
        req.add_header('X-Session-Id', session_id)
        req.add_header('X-Job-Type', job_type)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = response.read().decode('utf-8')
            print(f"[{session_id}] WEBHOOK DELIVERED: {res_data}")
            JOB_PROGRESS[session_id] = {"progress": 100, "message": "Complete!"}
            
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
            
        HEATMAP_FACES = 8
        
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
                JOB_PROGRESS[session_id] = {"progress": 85, "message": "Compiling Geometric Variance Topologies into GLB Heatmap..."}
                print(f"[{session_id}] Thread: Compiling geometry...")
                
                proc2 = subprocess.Popen(cmd_heatmap, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc2.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                proc2.wait()
                
                if os.path.exists(final_heatmap_path):
                    JOB_PROGRESS[session_id] = {"progress": 95, "message": "Pushing Heatmap Model to Cloud Storage..."}
                    s3_client = get_s3_client()
                    
                    s3_key = f"heatmap_asset_{session_id}.glb"
                    print(f"[{session_id}] Uploading Heatmap to AWS '{config.AWS_S3_BUCKET_NAME}'...")
                    s3_client.upload_file(
                        final_heatmap_path,
                        config.AWS_S3_BUCKET_NAME,
                        s3_key,
                        ExtraArgs={'ContentType': 'model/gltf-binary'}
                    )
                    
                    import urllib.request
                    payload = json.dumps({
                        "s3_key": s3_key,
                        "job_type": "heatmap",
                        "session_id": session_id
                    }).encode('utf-8')
                    
                    print(f"[{session_id}] Upload complete. Passing webhook signal downstream...")
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
                        JOB_PROGRESS[session_id] = {"progress": 0, "message": f"Webhook delivery failed: {str(e)[:80]}"}
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
