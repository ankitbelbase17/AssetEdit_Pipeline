#!/bin/bash
# =============================================================================
# Hunyuan3D-2.1 Runner — Lightning AI Studio
#
# Pipeline per image:
#   1. Background removed via Hunyuan3D's built-in BiRefNet (inside Docker, once)
#   2. N generation iterations, each in its own subdirectory
#
# Usage:
#   bash run_hunyuan3d.sh --batch images/ -d user/hunyuan3d:latest
#   bash run_hunyuan3d.sh --batch images/ -d user/hunyuan3d:latest --textured
#   bash run_hunyuan3d.sh --batch images/ -d user/hunyuan3d:latest --both
#   bash run_hunyuan3d.sh input.jpg -d user/hunyuan3d:latest --both --num-outputs 20
#
# Output layout:
#   ~/hunyuan_data/outputs/
#   ├── cat/
#   │   ├── cat_nobg.png        ← background-removed input (cached)
#   │   ├── cat_1/
#   │   │   ├── input.png
#   │   │   ├── cat_shape.glb
#   │   │   └── cat_texture.glb  (if --both or --textured)
#   │   ├── cat_2/ ...
#   │   └── cat_20/
#   ├── dog/
#   │   └── ...
#   └── batch_summary.json
# =============================================================================

set -e

# ─── DEFAULTS ────────────────────────────────────────────────────────────────
DOCKER_IMAGE="${HUNYUAN3D_DOCKER_IMAGE:-your_username/hunyuan3d:latest}"
OUTPUT_DIR="${HUNYUAN3D_OUTPUT_DIR:-$HOME/hunyuan_data}"
CACHE_DIR="${HUNYUAN3D_CACHE_DIR:-$HOME/.cache/hunyuan3d}"
MODE="shape"          # shape | textured | both
NUM_OUTPUTS=20
VIEWS=6
RESOLUTION=512
QUIET=false
TEST_ONLY=false
NO_FALLBACK=false
INPUT_IMAGE=""
BATCH_FOLDER=""

# ─── COLORS ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ─── HELP ────────────────────────────────────────────────────────────────────
usage() {
    echo -e "${BOLD}Usage:${NC}"
    echo "  bash run_hunyuan3d.sh <input.jpg>    [options]   # single image"
    echo "  bash run_hunyuan3d.sh --batch <dir>  [options]   # folder of JPGs"
    echo ""
    echo -e "${BOLD}Mode (pick one):${NC}"
    echo "  (default)    shape only    → <stem>_shape.glb per iteration"
    echo "  --textured   texture only  → <stem>_texture.glb per iteration"
    echo "  --both       shape + texture each iteration"
    echo ""
    echo -e "${BOLD}Options:${NC}"
    echo "  -d, --docker-image  IMAGE   Docker Hub image (default: \$HUNYUAN3D_DOCKER_IMAGE)"
    echo "  -o, --output-dir    DIR     Output root (default: ~/hunyuan_data)"
    echo "  --cache-dir         DIR     Model weight cache (default: ~/.cache/hunyuan3d)"
    echo "  --num-outputs       N       Iterations per image (default: 20)"
    echo "  --views             N       Texture views (default: 6)"
    echo "  --resolution        N       Texture resolution px (default: 512)"
    echo "  --no-fallback               Don't fall back to shape if texture fails"
    echo "  --test                      Pre-flight checks only, no generation"
    echo "  -q, --quiet                 Suppress container output"
    echo "  -h, --help                  Show this help"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  bash run_hunyuan3d.sh --batch images/ -d myuser/hunyuan3d:latest"
    echo "  bash run_hunyuan3d.sh --batch images/ -d myuser/hunyuan3d:latest --both"
    echo "  bash run_hunyuan3d.sh --batch images/ -d myuser/hunyuan3d:latest --both --num-outputs 20"
    echo "  bash run_hunyuan3d.sh input.jpg -d myuser/hunyuan3d:latest --both --num-outputs 5"
    echo ""
    echo -e "${BOLD}Save your image name permanently:${NC}"
    echo "  echo 'export HUNYUAN3D_DOCKER_IMAGE=myuser/hunyuan3d:latest' >> ~/.bashrc && source ~/.bashrc"
}

# ─── PARSE ARGS ──────────────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then usage; exit 1; fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch|-b)       BATCH_FOLDER="$2";  shift 2 ;;
        -d|--docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
        -o|--output-dir)  OUTPUT_DIR="$2";    shift 2 ;;
        --cache-dir)      CACHE_DIR="$2";     shift 2 ;;
        --textured)       MODE="textured";    shift   ;;
        --both)           MODE="both";        shift   ;;
        --num-outputs)    NUM_OUTPUTS="$2";   shift 2 ;;
        --views)          VIEWS="$2";         shift 2 ;;
        --resolution)     RESOLUTION="$2";    shift 2 ;;
        --no-fallback)    NO_FALLBACK=true;   shift   ;;
        --test)           TEST_ONLY=true;     shift   ;;
        -q|--quiet)       QUIET=true;         shift   ;;
        -h|--help)        usage; exit 0       ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"; usage; exit 1 ;;
        *)
            # Positional = single input image
            INPUT_IMAGE="$1"; shift ;;
    esac
done

# ─── VALIDATE INPUT ───────────────────────────────────────────────────────────
if [[ -z "$BATCH_FOLDER" && -z "$INPUT_IMAGE" ]]; then
    echo -e "${RED}✗ Provide either a single image or --batch <folder>${NC}"
    usage; exit 1
fi
if [[ -n "$BATCH_FOLDER" && -n "$INPUT_IMAGE" ]]; then
    echo -e "${RED}✗ Use either a single image OR --batch, not both${NC}"
    exit 1
fi

# ─── HEADER ──────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}${CYAN}🚀 Hunyuan3D-2.1 Pipeline${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [[ -n "$BATCH_FOLDER" ]]; then
    echo -e "  ${GREEN}✓ Mode:${NC}         BATCH  →  $BATCH_FOLDER"
else
    echo -e "  ${GREEN}✓ Mode:${NC}         SINGLE →  $INPUT_IMAGE"
fi
echo -e "  ${GREEN}✓ Generation:${NC}   $(echo $MODE | tr '[:lower:]' '[:upper:]')"
echo -e "  ${GREEN}✓ Iterations:${NC}   $NUM_OUTPUTS per image"
echo -e "  ${GREEN}✓ Bg removal:${NC}   BiRefNet (Hunyuan3D built-in, runs once per image)"
echo -e "  ${GREEN}✓ Docker image:${NC} $DOCKER_IMAGE"
echo -e "  ${GREEN}✓ Output dir:${NC}   $OUTPUT_DIR/outputs/"
echo -e "  ${GREEN}✓ Model cache:${NC}  $CACHE_DIR  (reused across runs)"
[[ "$MODE" != "shape" ]] && \
    echo -e "  ${GREEN}✓ Texture:${NC}      ${VIEWS} views @ ${RESOLUTION}px"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# ─── DEPENDENCY CHECKS ───────────────────────────────────────────────────────
echo -e "${BOLD}[1/4] Checking dependencies...${NC}"

if ! command -v docker &>/dev/null; then
    echo -e "${RED}✗ Docker not found.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Docker:${NC}  $(docker --version | cut -d' ' -f3 | tr -d ',')"

if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✗ Python3 not found.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Python:${NC}  $(python3 --version)"

if command -v nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  ${GREEN}✓ GPU:${NC}     $GPU"
else
    echo -e "  ${YELLOW}⚠ No NVIDIA GPU detected.${NC}"
fi

# Locate runner script (same dir as this script, or current dir)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="$SCRIPT_DIR/hunyuan3d_runner.py"
[[ ! -f "$RUNNER" ]] && RUNNER="./hunyuan3d_runner.py"
if [[ ! -f "$RUNNER" ]]; then
    echo -e "${RED}✗ hunyuan3d_runner.py not found next to this script.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Runner:${NC}  $RUNNER"

# ─── DOCKER IMAGE ─────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[2/4] Checking Docker image...${NC}"
IMAGE_ID=$(docker images -q "$DOCKER_IMAGE" 2>/dev/null)
if [[ -n "$IMAGE_ID" ]]; then
    echo -e "  ${GREEN}✓ Found locally:${NC} $DOCKER_IMAGE"
else
    echo -e "  ${YELLOW}⚠ Not found locally — pulling from Docker Hub...${NC}"
    if docker pull "$DOCKER_IMAGE"; then
        echo -e "  ${GREEN}✓ Pull successful.${NC}"
    else
        echo -e "${RED}✗ Pull failed. Check name and run: docker login${NC}"
        exit 1
    fi
fi

# ─── MODEL CACHE ──────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[3/4] Preparing model cache...${NC}"
mkdir -p "$CACHE_DIR"
CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
CACHE_FILES=$(find "$CACHE_DIR" -type f 2>/dev/null | wc -l | tr -d ' ')
if [[ "$CACHE_FILES" -gt 0 ]]; then
    echo -e "  ${GREEN}✓ Cache exists:${NC} $CACHE_DIR  ($CACHE_SIZE, $CACHE_FILES files)"
    echo -e "  ${GREEN}✓ Model will NOT be re-downloaded.${NC}"
else
    echo -e "  ${YELLOW}⚠ Cache is empty:${NC} $CACHE_DIR"
    echo -e "  ${CYAN}  Model will be downloaded on first run and cached here.${NC}"
fi

# ─── INPUT CHECK ──────────────────────────────────────────────────────────────
echo -e "\n${BOLD}[4/4] Checking input...${NC}"
if [[ -n "$BATCH_FOLDER" ]]; then
    if [[ ! -d "$BATCH_FOLDER" ]]; then
        echo -e "${RED}✗ Batch folder not found: $BATCH_FOLDER${NC}"; exit 1
    fi
    IMG_COUNT=$(find "$BATCH_FOLDER" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l | tr -d ' ')
    if [[ "$IMG_COUNT" -eq 0 ]]; then
        echo -e "${RED}✗ No JPG images found in: $BATCH_FOLDER${NC}"; exit 1
    fi
    echo -e "  ${GREEN}✓ Batch folder:${NC} $BATCH_FOLDER  ($IMG_COUNT JPG images)"
    echo -e "  ${CYAN}  Total iterations planned: $IMG_COUNT × $NUM_OUTPUTS = $((IMG_COUNT * NUM_OUTPUTS))${NC}"
    [[ "$MODE" == "both" ]] && \
        echo -e "  ${CYAN}  Total files planned:      $((IMG_COUNT * NUM_OUTPUTS * 2)) (shape + texture each)${NC}"
else
    if [[ ! -f "$INPUT_IMAGE" ]]; then
        echo -e "${RED}✗ Input image not found: $INPUT_IMAGE${NC}"; exit 1
    fi
    echo -e "  ${GREEN}✓ Input image:${NC} $INPUT_IMAGE"
fi

# ─── TEST MODE EARLY EXIT ─────────────────────────────────────────────────────
if [[ "$TEST_ONLY" == true ]]; then
    echo -e "\n${GREEN}✅ Pre-flight passed. Ready to generate (--test mode, stopping here).${NC}\n"
    exit 0
fi

# ─── BUILD PYTHON COMMAND ─────────────────────────────────────────────────────
PYTHON_ARGS=(
    "$RUNNER"
    "--docker-image" "$DOCKER_IMAGE"
    "--output-dir"   "$OUTPUT_DIR"
    "--cache-dir"    "$CACHE_DIR"
    "--num-outputs"  "$NUM_OUTPUTS"
    "--views"        "$VIEWS"
    "--resolution"   "$RESOLUTION"
)

# Input: batch or single
[[ -n "$BATCH_FOLDER" ]] && PYTHON_ARGS+=("--batch" "$BATCH_FOLDER")
[[ -n "$INPUT_IMAGE"  ]] && PYTHON_ARGS+=("$INPUT_IMAGE")

# Mode flags
[[ "$MODE" == "textured" ]] && PYTHON_ARGS+=("--textured")
[[ "$MODE" == "both"     ]] && PYTHON_ARGS+=("--both")

# Optional flags
[[ "$NO_FALLBACK" == true ]] && PYTHON_ARGS+=("--no-fallback")
[[ "$QUIET"       == true ]] && PYTHON_ARGS+=("--quiet")

# ─── RUN ─────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}Starting generation...${NC}"
echo -e "  ${CYAN}python3 ${PYTHON_ARGS[*]}${NC}\n"

START_TIME=$(date +%s)

python3 "${PYTHON_ARGS[@]}"
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

# ─── RESULT ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

OUTPUT_ROOT="$OUTPUT_DIR/outputs"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✅ Done in ${MINS}m ${SECS}s${NC}"
    echo ""

    # Count generated GLB files
    GLB_COUNT=$(find "$OUTPUT_ROOT" -name "*.glb" 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${BOLD}Results in $OUTPUT_ROOT:${NC}  ($GLB_COUNT .glb files total)"

    # Show per-image subdirectory summary
    for img_dir in "$OUTPUT_ROOT"/*/; do
        [[ -d "$img_dir" ]] || continue
        img_name=$(basename "$img_dir")
        iter_count=$(find "$img_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
        glb_count=$(find "$img_dir" -name "*.glb" 2>/dev/null | wc -l | tr -d ' ')
        nobg_exists=$(find "$img_dir" -maxdepth 1 -name "*_nobg.png" 2>/dev/null | wc -l | tr -d ' ')
        nobg_tag=""
        [[ "$nobg_exists" -gt 0 ]] && nobg_tag="  🔲 bg removed"
        echo -e "  📁 ${GREEN}$img_name/${NC}  →  $iter_count iteration dirs,  $glb_count .glb files${nobg_tag}"
    done

    # Summary JSON
    SUMMARY="$OUTPUT_ROOT/batch_summary.json"
    [[ -f "$SUMMARY" ]] && echo -e "\n  📋 Summary: $SUMMARY"

    echo ""
    echo -e "${CYAN}Tip: Drag any .glb to https://gltf-viewer.donmccurdy.com to preview.${NC}"
else
    echo -e "${RED}${BOLD}❌ Pipeline failed (exit code: $EXIT_CODE)${NC}"
    echo ""
    echo "  Common fixes:"
    echo "    • GPU OOM      → lower --resolution (e.g. 256) or --views (e.g. 4)"
    echo "    • No GPU       → make sure Lightning AI studio has a GPU attached"
    echo "    • Docker auth  → run: docker login"
    echo "    • Wrong image  → check: docker pull $DOCKER_IMAGE"
fi

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
exit $EXIT_CODE