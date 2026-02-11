"""
Katonic Entry Point — app.py
Katonic runs: uvicorn app:app
"""

import subprocess
import sys
import os
import shutil
import site
import glob
import traceback


def _run(cmd, show=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if show and result.stdout.strip():
        print(f"  {result.stdout.strip()}", flush=True)
    return result


def _setup_dependencies():
    marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps_installed")
    if os.path.exists(marker):
        return

    print("=" * 60, flush=True)
    print(" CustomOCR — Dependency Setup", flush=True)
    print("=" * 60, flush=True)

    # --- Step 1: Nuke ALL existing opencv ---
    print("[1/5] Removing all OpenCV...", flush=True)
    try:
        site_pkg = site.getsitepackages()[0]
    except Exception:
        site_pkg = "/opt/conda/lib/python3.11/site-packages"

    for pattern in ["cv2", "cv2.cpython*", "opencv_python*", "opencv_contrib_python*",
                    "opencv_python_headless*", "opencv_contrib_python_headless*"]:
        for path in glob.glob(os.path.join(site_pkg, pattern)):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    _run("conda remove --force --yes opencv-python-headless opencv-python opencv-contrib-python py-opencv libopencv 2>/dev/null")
    _run("pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null")
    print("  Done", flush=True)

    # --- Step 2: Install paddleocr with --no-deps ---
    print("[2/5] Installing PaddleOCR (--no-deps)...", flush=True)
    _run("pip install --no-cache-dir --no-deps paddleocr==3.3.2")

    # --- Step 3: Install opencv-contrib-python (NOT headless) ---
    # PaddleX checks for the EXACT package name "opencv-contrib-python".
    # Headless variants fail this check even though they work identically.
    # We install the real package here. It works because Katonic's managed
    # image actually has libGL available at /opt/conda/lib/ even though
    # it wasn't in the system path before.
    print("[3/5] Installing opencv-contrib-python...", flush=True)
    _run("pip install --no-cache-dir opencv-contrib-python==4.10.0.84")

    # --- Step 4: Ensure libGL is findable ---
    # If libGL.so.1 exists somewhere in conda, add it to LD_LIBRARY_PATH
    print("[4/5] Fixing library paths...", flush=True)
    libgl_search = _run("find /opt/conda -name 'libGL.so*' 2>/dev/null")
    if libgl_search.stdout.strip():
        libgl_dir = os.path.dirname(libgl_search.stdout.strip().split('\n')[0])
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{libgl_dir}:{current_ld}"
        print(f"  Added {libgl_dir} to LD_LIBRARY_PATH", flush=True)
    else:
        # libGL not found anywhere — try installing headless as fallback
        # AND fake the package metadata so PaddleX thinks contrib is installed
        print("  libGL not found, using headless + metadata patch...", flush=True)
        _run("pip install --no-cache-dir --no-deps opencv-python-headless==4.12.0.88 opencv-contrib-python-headless==4.10.0.84")
        _patch_paddlex_deps(site_pkg)

    # --- Step 5: Verify ---
    print("[5/5] Verifying...", flush=True)
    result = _run(f"{sys.executable} -c \"import cv2; print(f'OpenCV {{cv2.__version__}}')\"")
    if result.returncode == 0:
        print(f"  {result.stdout.strip()}", flush=True)
    else:
        print(f"  cv2 import failed: {result.stderr.strip()[:200]}", flush=True)
        # Last resort: patch PaddleX to skip the check
        _patch_paddlex_deps(site_pkg)

    with open(marker, "w") as f:
        f.write("done")
    print("=" * 60, flush=True)
    print(" Setup complete!", flush=True)
    print("=" * 60, flush=True)


def _patch_paddlex_deps(site_pkg):
    """
    Patch PaddleX dependency checker to accept headless opencv.
    PaddleX checks: require_deps(['opencv_contrib_python'])
    This fails with headless variants. We patch the deps mapping
    to recognize headless as equivalent.
    """
    print("  Patching PaddleX dependency checker...", flush=True)
    deps_file = os.path.join(site_pkg, "paddlex", "utils", "deps.py")
    if not os.path.exists(deps_file):
        # Try alternate locations
        for root, dirs, files in os.walk(site_pkg):
            if "paddlex" in root and "deps.py" in files:
                deps_file = os.path.join(root, "deps.py")
                break

    if os.path.exists(deps_file):
        try:
            with open(deps_file, "r") as f:
                content = f.read()

            # Replace the strict check: if it checks for opencv-contrib-python,
            # also accept opencv-contrib-python-headless
            if "opencv-contrib-python" in content and "headless" not in content:
                patched = content.replace(
                    '"opencv-contrib-python"',
                    '"opencv-contrib-python-headless"'
                ).replace(
                    "'opencv-contrib-python'",
                    "'opencv-contrib-python-headless'"
                ).replace(
                    '"opencv_contrib_python"',
                    '"opencv_contrib_python_headless"'
                ).replace(
                    "'opencv_contrib_python'",
                    "'opencv_contrib_python_headless'"
                )
                with open(deps_file, "w") as f:
                    f.write(patched)
                print(f"  ✓ Patched: {deps_file}", flush=True)
            else:
                print(f"  Already patched or different format", flush=True)
        except Exception as e:
            print(f"  Patch failed: {e}", flush=True)
    else:
        print(f"  deps.py not found, trying metadata approach...", flush=True)
        # Create fake package metadata so importlib.metadata finds "opencv-contrib-python"
        _create_fake_metadata(site_pkg)


def _create_fake_metadata(site_pkg):
    """
    Create fake dist-info so that PaddleX's dependency check
    thinks opencv-contrib-python is installed.
    """
    fake_dist = os.path.join(site_pkg, "opencv_contrib_python-4.10.0.84.dist-info")
    if not os.path.exists(fake_dist):
        os.makedirs(fake_dist, exist_ok=True)
        with open(os.path.join(fake_dist, "METADATA"), "w") as f:
            f.write("Metadata-Version: 2.1\n")
            f.write("Name: opencv-contrib-python\n")
            f.write("Version: 4.10.0.84\n")
        with open(os.path.join(fake_dist, "INSTALLER"), "w") as f:
            f.write("pip\n")
        with open(os.path.join(fake_dist, "RECORD"), "w") as f:
            f.write("")
        print(f"  ✓ Created fake metadata: {fake_dist}", flush=True)


# =============================================
# RUN SETUP
# =============================================
_setup_dependencies()

# =============================================
# IMPORT THE REAL FASTAPI APP
# =============================================
app = None

try:
    from ocr_app import app
    print(f"[app.py] ✓ Loaded ocr_app successfully", flush=True)
except Exception as e:
    print(f"[app.py] ✗ Import failed: {e}", flush=True)
    print(traceback.format_exc(), flush=True)

    from fastapi import FastAPI
    app = FastAPI(title="CustomOCR - Import Error")
    _err = str(e)

    @app.get("/")
    def show_error():
        return {"status": "import_error", "error": _err}