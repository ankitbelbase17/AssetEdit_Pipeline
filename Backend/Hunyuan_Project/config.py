import os

# Backend Configurations for the 3D generation server

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# Path to the shell script that wraps the Docker execution
RUNNER_SCRIPT = os.path.join(BASE_DIR, "run_hunyuan3d.sh")

# Server settings
HOST = "0.0.0.0"
PORT = 8000

# Cloudflare Tunnel Configuration
# When True, the server will attempt to start a cloudflare tunnel and expose a public URL
USE_CLOUDFLARE_TUNNEL = True

# Runner Script Default Parameters
DEFAULT_MODE = "--textured"  # Default to texture as per user spec
DOCKER_IMAGE = "belbaseankit17/hunyuan_custom_fixed:v4"

DEFAULT_RESOLUTION = 512
DEFAULT_VIEWS = 6
NUM_OUTPUTS = 1  # For API usage, we usually just want 1 iteration per request

# ==============================================================================
# AWS S3 STORAGE CONFIGURATION
# ==============================================================================
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION_NAME = ""
AWS_S3_BUCKET_NAME = "si3dr-3d-assets-bucket"  # Auto-created if it doesn't exist

