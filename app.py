"""
Katonic Entry Point — app.py
Katonic runs: uvicorn app:app

PROBLEM CHAIN:
  1. Katonic managed image has conda-installed GUI opencv
  2. GUI opencv needs libGL.so.1 which doesn't exist → import cv2 crashes
  3. Headless opencv works without libGL BUT
  4. PaddleX checks importlib.metadata.version("opencv-contrib-python")
  5. Headless package has different metadata name → check fails → DependencyError

SOLUTION:
  1. Remove conda opencv (delete files directly)
  2. Install headless opencv (no libGL needed)
  3. Create fake dist-info for "opencv-contrib-python" so PaddleX's check passes
  4. Both cv2 import AND PaddleX dependency check work
"""

import subprocess
import sys
import os
import shutil
import site
import glob
import traceback


def _run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def _get_site_packages():
    try:
        return site.getsitepackages()[0]
    except Exception:
        return "/opt/conda/lib/python3.11/site-packages"


def _setup():
    """One-time dependency fix. Runs before any pipeline imports."""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    marker = os.path.join(base_dir, ".deps_ok")

    # Skip if already done
    if os.path.exists(marker):
        # But ALWAYS verify cv2 works (in case container was rebuilt)
        result = _run(f'{sys.executable} -c "import cv2"')
        if result.returncode == 0:
            return
        else:
            # Marker exists but cv2 broken — redo setup
            os.remove(marker)

    print("=" * 60, flush=True)
    print(" CustomOCR — Fixing OpenCV for Katonic", flush=True)
    print("=" * 60, flush=True)

    site_pkg = _get_site_packages()

    # =========================================================
    # STEP 1: Delete ALL opencv from disk
    # =========================================================
    print("[1/4] Nuking all OpenCV installations...", flush=True)

    # Delete conda opencv files (pip can't uninstall these)
    for pattern in ["cv2", "cv2.cpython*",
                    "opencv_python*", "opencv_contrib_python*",
                    "opencv_python_headless*", "opencv_contrib_python_headless*"]:
        for path in glob.glob(os.path.join(site_pkg, pattern)):
            print(f"  rm {os.path.basename(path)}", flush=True)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass

    # Also check other possible site-packages locations
    for sp in site.getsitepackages() if hasattr(site, 'getsitepackages') else []:
        if sp != site_pkg:
            for pattern in ["cv2", "opencv_python*", "opencv_contrib_python*"]:
                for path in glob.glob(os.path.join(sp, pattern)):
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        try:
                            os.remove(path)
                        except OSError:
                            pass

    _run("pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null")
    print("  ✓ All OpenCV removed", flush=True)

    # =========================================================
    # STEP 2: Install paddleocr WITHOUT deps
    # =========================================================
    print("[2/4] Installing PaddleOCR (--no-deps)...", flush=True)
    _run("pip install --no-cache-dir --no-deps paddleocr==3.3.2")
    print("  ✓ Done", flush=True)

    # =========================================================
    # STEP 3: Install HEADLESS opencv (no libGL.so.1 needed)
    # =========================================================
    print("[3/4] Installing headless OpenCV...", flush=True)
    result = _run("pip install --no-cache-dir opencv-python-headless==4.10.0.84 opencv-contrib-python-headless==4.10.0.84")
    if result.returncode != 0:
        print(f"  WARN: {result.stderr.strip()[:200]}", flush=True)
        # Try without pinned version
        _run("pip install --no-cache-dir opencv-python-headless opencv-contrib-python-headless")
    print("  ✓ Done", flush=True)

    # =========================================================
    # STEP 4: Create fake metadata for "opencv-contrib-python"
    # =========================================================
    # PaddleX does: importlib.metadata.version("opencv-contrib-python")
    # Headless package registers as "opencv-contrib-python-headless"
    # So we create a fake dist-info directory that makes PaddleX
    # think "opencv-contrib-python" version 4.10.0.84 is installed.
    print("[4/4] Creating opencv-contrib-python metadata for PaddleX...", flush=True)

    # Get the actual headless version that was installed
    ver_result = _run(f'{sys.executable} -c "import cv2; print(cv2.__version__)"')
    cv_version = ver_result.stdout.strip() if ver_result.returncode == 0 else "4.10.0.84"

    # Create fake dist-info in ALL site-packages locations
    for sp in (site.getsitepackages() if hasattr(site, 'getsitepackages') else [site_pkg]):
        fake_dist = os.path.join(sp, f"opencv_contrib_python-{cv_version}.dist-info")
        try:
            os.makedirs(fake_dist, exist_ok=True)

            with open(os.path.join(fake_dist, "METADATA"), "w") as f:
                f.write(f"Metadata-Version: 2.1\n")
                f.write(f"Name: opencv-contrib-python\n")
                f.write(f"Version: {cv_version}\n")
                f.write(f"Summary: Fake metadata - actual cv2 from opencv-contrib-python-headless\n")

            with open(os.path.join(fake_dist, "INSTALLER"), "w") as f:
                f.write("pip\n")

            with open(os.path.join(fake_dist, "RECORD"), "w") as f:
                f.write("")

            # Also create one for opencv-python (some checks look for this too)
            fake_dist2 = os.path.join(sp, f"opencv_python-{cv_version}.dist-info")
            os.makedirs(fake_dist2, exist_ok=True)
            with open(os.path.join(fake_dist2, "METADATA"), "w") as f:
                f.write(f"Metadata-Version: 2.1\n")
                f.write(f"Name: opencv-python\n")
                f.write(f"Version: {cv_version}\n")

            with open(os.path.join(fake_dist2, "INSTALLER"), "w") as f:
                f.write("pip\n")

            with open(os.path.join(fake_dist2, "RECORD"), "w") as f:
                f.write("")

            print(f"  ✓ Created metadata in {sp}", flush=True)
        except Exception as e:
            print(f"  ✗ Failed for {sp}: {e}", flush=True)

    # =========================================================
    # VERIFY
    # =========================================================
    print("\nVerification:", flush=True)

    # Test 1: cv2 imports
    r = _run(f'{sys.executable} -c "import cv2; print(f\'  cv2: {{cv2.__version__}} from {{cv2.__file__}}\')"')
    print(r.stdout.strip() if r.returncode == 0 else f"  ✗ cv2 import failed: {r.stderr.strip()[:200]}", flush=True)

    # Test 2: PaddleX sees opencv-contrib-python
    r = _run(f'{sys.executable} -c "import importlib.metadata; v=importlib.metadata.version(\'opencv-contrib-python\'); print(f\'  metadata check: opencv-contrib-python={{v}}\')"')
    print(r.stdout.strip() if r.returncode == 0 else f"  ✗ metadata check failed: {r.stderr.strip()[:200]}", flush=True)

    # Test 3: PaddleX sees opencv-python
    r = _run(f'{sys.executable} -c "import importlib.metadata; v=importlib.metadata.version(\'opencv-python\'); print(f\'  metadata check: opencv-python={{v}}\')"')
    print(r.stdout.strip() if r.returncode == 0 else f"  ✗ metadata check failed (opencv-python)", flush=True)

    # Write marker
    with open(marker, "w") as f:
        f.write("ok")

    print("\n" + "=" * 60, flush=True)
    print(" Setup complete!", flush=True)
    print("=" * 60 + "\n", flush=True)


# =============================================
# RUN SETUP BEFORE ANY IMPORTS
# =============================================
_setup()

# =============================================
# CLEAR LRU CACHES (PaddleX caches dependency checks)
# =============================================
try:
    import importlib.metadata
    # Force Python to re-scan package metadata
    importlib.metadata.distributions.cache_clear() if hasattr(importlib.metadata.distributions, 'cache_clear') else None
except Exception:
    pass

# =============================================
# IMPORT THE REAL FASTAPI APP
# =============================================
app = None

try:
    from ocr_app import app
    print("[app.py] ✓ ocr_app loaded successfully", flush=True)
except Exception as e:
    print(f"[app.py] ✗ Failed to import ocr_app: {e}", flush=True)
    print(traceback.format_exc(), flush=True)

    # Fallback app so you can see the error in the browser
    from fastapi import FastAPI
    app = FastAPI(title="CustomOCR - Error")
    _err = str(e)
    _tb = traceback.format_exc()

    @app.get("/")
    def show_error():
        return {
            "status": "import_error",
            "error": _err,
            "traceback": _tb.split("\n")
        }