"""
Katonic Entry Point — app.py
Katonic runs: uvicorn app:app

FIXES:
  1. OpenCV headless + fake metadata for PaddleX
  2. LibreOffice for .docx/.pptx/.xlsx conversion
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
        result = _run(f'{sys.executable} -c "import cv2"')
        soffice = _run("which soffice")
        if result.returncode == 0 and soffice.returncode == 0:
            return
        else:
            os.remove(marker)

    print("=" * 60, flush=True)
    print(" CustomOCR — Fixing Dependencies for Katonic", flush=True)
    print("=" * 60, flush=True)

    site_pkg = _get_site_packages()

    # =========================================================
    # STEP 1: Delete ALL opencv from disk
    # =========================================================
    print("[1/5] Removing all OpenCV installations...", flush=True)

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
    print("  ✓ Done", flush=True)

    # =========================================================
    # STEP 2: Install paddleocr WITHOUT deps
    # =========================================================
    print("[2/5] Installing PaddleOCR (--no-deps)...", flush=True)
    _run("pip install --no-cache-dir --no-deps paddleocr==3.3.2")
    print("  ✓ Done", flush=True)

    # =========================================================
    # STEP 3: Install HEADLESS opencv
    # =========================================================
    print("[3/5] Installing headless OpenCV...", flush=True)
    result = _run("pip install --no-cache-dir opencv-python-headless==4.10.0.84 opencv-contrib-python-headless==4.10.0.84")
    if result.returncode != 0:
        _run("pip install --no-cache-dir opencv-python-headless opencv-contrib-python-headless")
    print("  ✓ Done", flush=True)

    # =========================================================
    # STEP 4: Create fake metadata for PaddleX
    # =========================================================
    print("[4/5] Creating opencv metadata for PaddleX...", flush=True)

    ver_result = _run(f'{sys.executable} -c "import cv2; print(cv2.__version__)"')
    cv_version = ver_result.stdout.strip() if ver_result.returncode == 0 else "4.10.0.84"

    for sp in (site.getsitepackages() if hasattr(site, 'getsitepackages') else [site_pkg]):
        for pkg_name in ["opencv-contrib-python", "opencv-python"]:
            dist_name = pkg_name.replace("-", "_")
            fake_dist = os.path.join(sp, f"{dist_name}-{cv_version}.dist-info")
            try:
                os.makedirs(fake_dist, exist_ok=True)
                with open(os.path.join(fake_dist, "METADATA"), "w") as f:
                    f.write(f"Metadata-Version: 2.1\nName: {pkg_name}\nVersion: {cv_version}\n")
                with open(os.path.join(fake_dist, "INSTALLER"), "w") as f:
                    f.write("pip\n")
                with open(os.path.join(fake_dist, "RECORD"), "w") as f:
                    f.write("")
            except Exception as e:
                print(f"  ✗ {fake_dist}: {e}", flush=True)
    print("  ✓ Done", flush=True)

    # =========================================================
    # STEP 5: Install LibreOffice (for .docx/.pptx/.xlsx)
    # =========================================================
    print("[5/5] Installing LibreOffice...", flush=True)

    soffice_check = _run("which soffice")
    if soffice_check.returncode == 0:
        print(f"  ✓ Already installed: {soffice_check.stdout.strip()}", flush=True)
    else:
        result = _run(
            "apt-get update -qq && "
            "apt-get install -y -qq --no-install-recommends "
            "apt-get install libreoffice-writer libreoffice-calc libreoffice-impress"
            
        )
        if result.returncode != 0:
            print("  Minimal install failed, trying full LibreOffice...", flush=True)
            result = _run("apt-get install -y -qq libreoffice")

        if result.returncode != 0:
            print("  apt failed, trying conda...", flush=True)
            result = _run("conda install -y -c conda-forge libreoffice")

        verify = _run("which soffice")
        if verify.returncode == 0:
            print(f"  ✓ Installed: {verify.stdout.strip()}", flush=True)
        else:
            print("  ✗ LibreOffice install failed. .docx/.pptx/.xlsx will not work.", flush=True)
            print(f"  Error: {result.stderr.strip()[:300]}", flush=True)

    # =========================================================
    # VERIFY ALL
    # =========================================================
    print("\nVerification:", flush=True)

    r = _run(f'{sys.executable} -c "import cv2; print(cv2.__version__)"')
    print(f"  cv2: {r.stdout.strip()}" if r.returncode == 0 else "  ✗ cv2 failed", flush=True)

    r = _run(f'{sys.executable} -c "import importlib.metadata; print(importlib.metadata.version(\'opencv-contrib-python\'))"')
    print(f"  opencv-contrib-python metadata: {r.stdout.strip()}" if r.returncode == 0 else "  ✗ metadata failed", flush=True)

    r = _run("soffice --version")
    print(f"  soffice: {r.stdout.strip()}" if r.returncode == 0 else "  ✗ soffice not found", flush=True)

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
# CLEAR LRU CACHES
# =============================================
try:
    import importlib.metadata
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