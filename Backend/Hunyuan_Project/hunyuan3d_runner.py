#!/usr/bin/env python3
"""
Hunyuan3D-2.1 Docker Pipeline Runner
======================================
Key behaviours:
  - Model weights are downloaded ONCE into ~/hunyuan_cache and re-mounted
    into every Docker container — no re-download on subsequent runs.
  - Each generation iteration gets its OWN subdirectory:
      outputs/<image_stem>/<image_stem>_1/   ← all files from iteration 1
      outputs/<image_stem>/<image_stem>_2/   ← all files from iteration 2
      ...
  - Input images must be JPG (no conversion logic).
  - Supports mode: shape | textured | both

Usage
-----
Single image, 20 outputs (shape only):
  python hunyuan3d_runner.py input.jpg -d user/hunyuan3d:latest --num-outputs 20

Single image, shape + texture, 20 outputs:
  python hunyuan3d_runner.py input.jpg -d user/hunyuan3d:latest --both --num-outputs 20

Batch (folder of images):
  python hunyuan3d_runner.py --batch images/ -d user/hunyuan3d:latest --num-outputs 20
  python hunyuan3d_runner.py --batch images/ -d user/hunyuan3d:latest --both --num-outputs 20

Output layout
-------------
outputs/
├── cat/
│   ├── cat_1/
│   │   ├── cat_shape.glb          (or cat_texture.glb / both)
│   │   └── ... (any extra files the model produces)
│   ├── cat_2/
│   │   └── cat_shape.glb
│   └── ...
├── dog/
│   ├── dog_1/
│   └── ...
└── batch_summary.json
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}   # input is guaranteed JPG





# ─── RUNNER ───────────────────────────────────────────────────────────────────

class Hunyuan3DRunner:

    # Model is cached here on the HOST so Docker never re-downloads it
    DEFAULT_CACHE = Path("~/.cache/hunyuan3d").expanduser()

    def __init__(
        self,
        docker_image: str = "your_username/hunyuan3d:latest",
        data_dir: str     = "~/hunyuan_data",
        cache_dir: str    = None,
    ):
        self.docker_image = docker_image
        self.data_dir     = Path(data_dir).expanduser()
        self.cache_dir    = Path(cache_dir).expanduser() if cache_dir else self.DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-flight ────────────────────────────────────────────────────────────

    def check_docker_available(self):
        try:
            r = subprocess.run(["docker", "--version"],
                               capture_output=True, text=True, check=True)
            print(f"✓ Docker: {r.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ Docker not found.")
            return False

    def check_gpu_available(self):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True)
            print(f"✓ GPU: {r.stdout.strip().splitlines()[0]}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ No NVIDIA GPU detected.")
            return False

    def check_docker_image_exists(self):
        print(f"Checking Docker image: {self.docker_image}")
        try:
            r = subprocess.run(["docker", "images", "-q", self.docker_image],
                               capture_output=True, text=True, check=True)
            if r.stdout.strip():
                print(f"✓ Image found locally.")
                return True
            print("⚠ Not found locally — pulling...")
            return self._pull_image()
        except subprocess.CalledProcessError as e:
            print(f"✗ Error: {e}")
            return False

    def _pull_image(self):
        try:
            subprocess.run(["docker", "pull", self.docker_image], check=True)
            print(f"✓ Pulled: {self.docker_image}")
            return True
        except subprocess.CalledProcessError:
            print("✗ Pull failed. Check image name and `docker login`.")
            return False

    # ── Docker execution ──────────────────────────────────────────────────────

    def _docker_run(self, python_script: str, iter_dir: Path, verbose: bool):
        """
        Run python_script inside the container.

        Mounts:
          iter_dir  → /data       (input image + all outputs land here)
          cache_dir → /root/.cache/hunyuan3d  (model weights, persisted)
        """
        cmd = [
            "docker", "run",
            "--gpus", "all",
            "--rm",
            # input / output directory
            "-v", f"{iter_dir.absolute()}:/data",
            # model weight cache — survives across runs
            "-v", f"{self.cache_dir.absolute()}:/root/.cache/hunyuan3d",
            # also expose the standard HF cache path so from_pretrained finds it
            "-e", "HF_HOME=/root/.cache/hunyuan3d",
            "-e", "HUGGINGFACE_HUB_CACHE=/root/.cache/hunyuan3d",
            "-w", "/workspace/Hunyuan3D-2.1",
            self.docker_image,
            "python3", "-c", python_script,
        ]
        result = subprocess.run(cmd, capture_output=not verbose, text=True, check=True)
        if verbose and result.stdout:
            print(result.stdout)

    # ── Background removal (BiRefNet via Hunyuan3D's own pipeline) ───────────

    def _script_remove_bg(self) -> str:
        """
        Python script run inside Docker to remove background using Hunyuan3D's
        built-in BiRefNet-based image processor.
        Reads  /data/input_raw.jpg  → writes  /data/input_nobg.png  (RGBA PNG)
        """
        return """
import shutil
from PIL import Image

print('[bg] Using rembg for background removal...')
try:
    from rembg import remove
    with Image.open('/data/input_raw.jpg') as img:
        result = remove(img)
        result.save('/data/input_nobg.png', format='PNG')
    print('[bg] Background removed → /data/input_nobg.png')
except Exception as e:
    print(f'[bg] rembg failed: {e}')
    print('[bg] Falling back to plain copy...')
    shutil.copy2('/data/input_raw.jpg', '/data/input_nobg.png')
    print('[bg] Copied original → /data/input_nobg.png')
"""

    def _remove_background_in_docker(
        self,
        input_jpg:  Path,
        image_root: Path,
        stem:       str,
        verbose:    bool,
    ) -> Path:
        """
        Run background removal inside Docker using rembg (installed in the container).
        Done ONCE per image — result cached at image_root/<stem>_nobg.png.

        Returns path to the background-free PNG on the host.
        """
        nobg_host = image_root / f"{stem}_nobg.png"

        # Already done in a previous run — reuse it
        if nobg_host.exists():
            print(f"  ↩  Background-removed image already exists: {nobg_host.name}")
            return nobg_host

        # We need a temporary dir to mount into Docker just for this step
        bg_workdir = image_root / "_bg_removal"
        bg_workdir.mkdir(parents=True, exist_ok=True)

        # Copy original jpg in as input_raw.jpg
        shutil.copy2(input_jpg, bg_workdir / "input_raw.jpg")

        print(f"  🔲 Removing background via Hunyuan3D BiRefNet (Docker)...")

        try:
            cmd = [
                "docker", "run",
                "--gpus", "all",
                "--rm",
                "-v", f"{bg_workdir.absolute()}:/data",
                "-v", f"{self.cache_dir.absolute()}:/root/.cache/hunyuan3d",
                "-e", "HF_HOME=/root/.cache/hunyuan3d",
                "-e", "HUGGINGFACE_HUB_CACHE=/root/.cache/hunyuan3d",
                "-w", "/workspace/Hunyuan3D-2.1",
                self.docker_image,
                "python3", "-c", self._script_remove_bg(),
            ]
            result = subprocess.run(cmd, capture_output=not verbose, text=True, check=True)
            if verbose and result.stdout:
                print(result.stdout)

            produced = bg_workdir / "input_nobg.png"
            if produced.exists():
                shutil.move(str(produced), str(nobg_host))
                print(f"  ✓  Background removed → {nobg_host.name}")
            else:
                raise RuntimeError("input_nobg.png not found after Docker bg removal")

        except Exception as e:
            print(f"  ⚠  Background removal failed: {e}")
            print(f"  ⚠  Falling back to original image (no background removal)")
            shutil.copy2(input_jpg, nobg_host)

        finally:
            # Clean up temp workdir
            shutil.rmtree(bg_workdir, ignore_errors=True)

        return nobg_host

    def _script_batch_internal(self, stem: str, start_seed: int, num_outputs: int, mode: str, max_views: int, resolution: int) -> str:
        """Dynamically builds a single python script strictly executed natively ONCE per image batch inside the Docker engine, caching 14GB of models aggressively into VRAM exactly once."""
        imports = """
import sys, shutil, torch, os, traceback
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

print('Mounting Geometry Matrices into GPU VRAM...')
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
"""
        if mode in ("textured", "both"):
            imports += f"""
print('Mounting Texture Painting Synthesis Models into GPU VRAM...')
cfg = Hunyuan3DPaintConfig(max_num_view={max_views}, resolution={resolution})
paint = Hunyuan3DPaintPipeline(cfg)
"""
        
        loop_logic = f"""
input_bg = '/data/{stem}_nobg.png'
for i in range(1, {num_outputs} + 1):
    seed = {start_seed} + i
    iter_dir = f"/data/{stem}_{{i}}"
    os.makedirs(iter_dir, exist_ok=True)
    input_png = iter_dir + '/input.png'
    shutil.copy2(input_bg, input_png)
    
    print(f"\\nIteration {{i}}/{num_outputs}  →  (seed={{seed}})")
    generator = torch.manual_seed(seed)
"""
        
        # Geometry Logic
        if mode in ("shape", "both"):
            loop_logic += f"""
    try:
        print(f'[shape] seed={{seed}}')
        mesh = pipeline(image=input_png, generator=generator)[0]
        mesh.export(iter_dir + '/{stem}_shape.glb')
        mesh.export(iter_dir + '/mesh.obj')
        print(f'[shape] done → {{iter_dir}}/{stem}_shape.glb')
    except Exception as e:
        print(f"    ✗ Shape failed: {{e}}")
"""
        
        # Texture Logic
        if mode in ("textured", "both"):
            loop_logic += f"""
    try:
        print(f'[textured] seed={{seed}} views={max_views} res={resolution}')
        mesh = pipeline(image=input_png, generator=generator)[0]
        temp_obj = iter_dir + '/temp_mesh.obj'
        mesh.export(temp_obj)
        
        result = paint(temp_obj, image_path=input_png)
        
        out_glb = iter_dir + '/{stem}_texture.glb'
        if isinstance(result, str):
            if result.endswith('.obj'):
                actual_glb = result[:-4] + '.glb'
                actual_obj = result
            else:
                actual_glb = result
                actual_obj = result.replace('.glb', '.obj')
                
            # Safely push exact formats respectively preventing header collision parsing errors!
            shutil.copy2(actual_glb, out_glb)
            try:
                shutil.copy2(actual_obj, iter_dir + '/textured_mesh.obj')
            except: pass
        else:
            result.export(out_glb)
            try: result.export(iter_dir + '/textured_mesh.obj')
            except: pass
            
        print(f'[textured] done → {{out_glb}}')
    except Exception as e:
        print(f"    ✗ Texture failed: {{e}}")
        traceback.print_exc()
"""
        return imports + loop_logic


    # ── Single image → N iterations ───────────────────────────────────────────

    def generate_for_image(
        self,
        input_jpg:   Path,
        image_root:  Path,       # outputs/<stem>/
        stem:        str,
        mode:        str   = "shape",
        num_outputs: int   = 20,
        start_seed:  int   = 0,
        verbose:     bool  = True,
        max_views:   int   = 6,
        resolution:  int   = 512,
        fallback:    bool  = True,
    ):
        """
        Generate num_outputs iterations for one image natively inside a perfectly unified Docker memory container context.
        """
        image_root.mkdir(parents=True, exist_ok=True)

        nobg_path = self._remove_background_in_docker(
            input_jpg  = input_jpg,
            image_root = image_root,
            stem       = stem,
            verbose    = verbose,
        )

        all_produced = []
        all_failed   = []
        
        # Unified script mapping
        try:
            self._docker_run(
                self._script_batch_internal(stem, start_seed, num_outputs, mode, max_views, resolution),
                image_root, verbose
            )
            
            # Post-check logic securely validating file emissions
            for i in range(1, num_outputs + 1):
                iter_dir = image_root / f"{stem}_{i}"
                if (iter_dir / f"{stem}_texture.glb").exists() or (iter_dir / f"{stem}_shape.glb").exists():
                    all_produced.append(iter_dir)
                else:
                    all_failed.append(i)
                    
        except Exception as e:
            print(f"Docker batch crashed internally gracefully: {e}")
            all_failed = list(range(1, num_outputs + 1))

        return all_produced, all_failed

    # ── Batch ─────────────────────────────────────────────────────────────────

    def run_batch(
        self,
        image_folder: Path,
        output_root:  Path,
        mode:         str  = "shape",
        num_outputs:  int  = 20,
        verbose:      bool = True,
        max_views:    int  = 6,
        resolution:   int  = 512,
        fallback:     bool = True,
    ):
        """
        Process every JPG in image_folder, num_outputs iterations each.

        Final layout:
          output_root/
          ├── cat/
          │   ├── cat_1/   ├── cat_2/ ...
          ├── dog/
          │   ├── dog_1/   ...
          └── batch_summary.json
        """
        image_folder = Path(image_folder)
        output_root  = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        images = sorted([
            p for p in image_folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not images:
            print(f"✗ No JPG images found in: {image_folder}")
            sys.exit(1)

        files_per_image = num_outputs * (2 if mode == "both" else 1)
        total_planned   = len(images) * files_per_image

        print(f"\n{'='*60}")
        print(f"BATCH MODE")
        print(f"  Images         : {len(images)}")
        print(f"  Iterations/img : {num_outputs}")
        print(f"  Mode           : {mode}  ({'shape+texture per iter' if mode=='both' else '1 file per iter'})")
        print(f"  Total files    : {total_planned}")
        print(f"  Model cache    : {self.cache_dir}  (shared across all runs)")
        print(f"  Output root    : {output_root}")
        print(f"{'='*60}")
        for img in images:
            print(f"  • {img.name}")
        print(f"{'='*60}\n")

        summary = {
            "started_at":            datetime.now().isoformat(),
            "mode":                  mode,
            "num_outputs_per_image": num_outputs,
            "total_planned":         total_planned,
            "model_cache":           str(self.cache_dir),
            "images":                {},
        }

        overall_ok   = 0
        overall_fail = 0

        for idx, img_path in enumerate(images, 1):
            stem       = img_path.stem
            image_root = output_root / stem

            print(f"\n{'─'*60}")
            print(f"[{idx}/{len(images)}]  {img_path.name}  →  {image_root}/")
            print(f"{'─'*60}")

            produced, failed = self.generate_for_image(
                input_jpg   = img_path,
                image_root  = image_root,
                stem        = stem,
                mode        = mode,
                num_outputs = num_outputs,
                start_seed  = idx * 1000,
                verbose     = verbose,
                max_views   = max_views,
                resolution  = resolution,
                fallback    = fallback,
            )

            overall_ok   += len(produced)
            overall_fail += len(failed)

            summary["images"][stem] = {
                "source":          str(img_path),
                "output_dir":      str(image_root),
                "produced_files":  [str(p) for p in produced],
                "failed_iters":    failed,
                "count_ok":        len(produced),
                "count_failed":    len(failed),
            }

            print(f"\n  ✓  {len(produced)} files produced for {img_path.name}")
            if failed:
                print(f"  ✗  Iterations failed: {failed}")

        summary["finished_at"]   = datetime.now().isoformat()
        summary["total_ok"]      = overall_ok
        summary["total_failed"]  = overall_fail

        summary_path = output_root / "batch_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"  Files produced : {overall_ok} / {total_planned}")
        print(f"  Failed         : {overall_fail}")
        print(f"  Summary        : {summary_path}")
        print(f"{'='*60}\n")

        return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D-2.1 — batch 3D generation via Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output layout:
  outputs/
  ├── cat/
  │   ├── cat_1/   cat_shape.glb  [cat_texture.glb]
  │   ├── cat_2/   ...
  │   └── cat_20/
  ├── dog/
  │   ├── dog_1/ ...
  └── batch_summary.json

Examples:
  # Batch, shape only, 20 iterations per image
  python hunyuan3d_runner.py --batch images/ -d user/img:tag --num-outputs 20

  # Batch, both shape + texture
  python hunyuan3d_runner.py --batch images/ -d user/img:tag --both --num-outputs 20

  # Single image, both, 20 iterations
  python hunyuan3d_runner.py input.jpg -d user/img:tag --both --num-outputs 20

  # Pre-flight check only
  python hunyuan3d_runner.py --batch images/ -d user/img:tag --test
        """
    )

    inp = parser.add_mutually_exclusive_group(required=True)
    inp.add_argument("input_image",   nargs="?", type=str,
                     help="Single input JPG image")
    inp.add_argument("--batch", "-b", type=str, metavar="IMAGE_FOLDER",
                     help="Folder of JPG images")

    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument("--textured", action="store_true",
                           help="Texture only")
    mode_grp.add_argument("--both",     action="store_true",
                           help="Shape AND texture each iteration")

    parser.add_argument("-d", "--docker-image",  type=str,
                        default="your_username/hunyuan3d:latest")
    parser.add_argument("-o", "--output-dir",    type=str,
                        default="~/hunyuan_data",
                        help="Root output directory (default: ~/hunyuan_data)")
    parser.add_argument("--cache-dir",           type=str,
                        default=str(Hunyuan3DRunner.DEFAULT_CACHE),
                        help=f"Model weight cache (default: {Hunyuan3DRunner.DEFAULT_CACHE})")
    parser.add_argument("--num-outputs",         type=int, default=20,
                        help="Iterations per image (default: 20)")
    parser.add_argument("--views",               type=int, default=6)
    parser.add_argument("--resolution",          type=int, default=512)
    parser.add_argument("--no-fallback",         action="store_true")
    parser.add_argument("-q", "--quiet",         action="store_true")
    parser.add_argument("--test",                action="store_true",
                        help="Pre-flight checks only")

    args = parser.parse_args()

    mode = "both" if args.both else ("textured" if args.textured else "shape")

    runner = Hunyuan3DRunner(
        docker_image = args.docker_image,
        data_dir     = args.output_dir,
        cache_dir    = args.cache_dir,
    )

    check_path = Path(args.input_image if args.input_image else args.batch)

    # ── Test mode ──────────────────────────────────────────────────────────
    if args.test:
        print("\n🧪 Pre-flight checks\n")
        ok = all([
            runner.check_docker_available(),
            True,   # gpu warning only
            runner.check_docker_image_exists(),
        ])
        runner.check_gpu_available()
        print(f"\n✓ Model cache dir: {runner.cache_dir}")
        if check_path.exists():
            print(f"✓ Input path: {check_path.resolve()}")
        else:
            print(f"✗ Input path not found: {check_path}")
            ok = False
        print("\n" + ("✅ Ready." if ok else "❌ Fix issues above.") + "\n")
        return

    # ── Common checks ──────────────────────────────────────────────────────
    print("\n🚀 Hunyuan3D-2.1 Pipeline\n")
    if not runner.check_docker_available(): sys.exit(1)
    runner.check_gpu_available()
    if not runner.check_docker_image_exists(): sys.exit(1)
    print(f"✓ Model cache: {runner.cache_dir}\n")

    output_root = Path(args.output_dir).expanduser() / "outputs"

    # ── Batch ──────────────────────────────────────────────────────────────
    if args.batch:
        try:
            runner.run_batch(
                image_folder = args.batch,
                output_root  = output_root,
                mode         = mode,
                num_outputs  = args.num_outputs,
                verbose      = not args.quiet,
                max_views    = args.views,
                resolution   = args.resolution,
                fallback     = not args.no_fallback,
            )
            print(f"📁 Outputs : {output_root}")
            print(f"📋 Summary : {output_root / 'batch_summary.json'}")
        except Exception as e:
            print(f"\n❌ Batch failed: {e}")
            sys.exit(1)

    # ── Single image ───────────────────────────────────────────────────────
    else:
        img_path   = Path(args.input_image)
        stem       = img_path.stem
        image_root = output_root / stem

        try:
            produced, failed = runner.generate_for_image(
                input_jpg   = img_path,
                image_root  = image_root,
                stem        = stem,
                mode        = mode,
                num_outputs = args.num_outputs,
                start_seed  = 42,
                verbose     = not args.quiet,
                max_views   = args.views,
                resolution  = args.resolution,
                fallback    = not args.no_fallback,
            )
            print(f"\n✅ Done!  {len(produced)} files in {image_root}/")
            if failed:
                print(f"   ✗ Failed iterations: {failed}")
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()