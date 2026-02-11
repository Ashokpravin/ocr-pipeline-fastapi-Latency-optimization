import numpy as np
import json
import ast
import psutil
import logging
from pathlib import Path
import shutil
import os
import re


def get_vert_dist(v0, v1):
    n = len(v0)
    ds = np.zeros((n,n))
    for i in range(n):
        mt = np.abs(np.array([v0[i]-v0, v0[i]-v1, v1[i]-v0, v1[i]-v1]))
        ds[i,:] = np.min(mt, 0)
    return ds

def boxes_area(box: np.ndarray) -> np.ndarray:
    return (box[2] - box[0]) * (box[3] - box[1])

def boxes_stats(boxes_true: np.ndarray) -> np.ndarray:

    '''Compute Intersection Over Union (IoU) between two sets of bounding boxes - 
    'boxes_true' and 'boxes_detection'. Both sets of boxes are expected to be in '(x_min, y_min, x_max, y_max)' format.
    
    Args:
        boxes_true (np.ndarray): 2D.
        FINISH THIS LATER
        '''

    area = boxes_area(boxes_true.T)

    top_left = np.maximum(boxes_true[:, None, :2], boxes_true[:, :2])
    bottom_right = np.minimum(boxes_true[:, None, 2:], boxes_true[:, 2:])

    area_inter = np.prod(
        np.clip(bottom_right - top_left, a_min=0, a_max=None), axis=2)
    area_union = area[:, None] + area - area_inter
    return area, area_inter, area_union
    

def boxes_iou(boxes_true: np.ndarray, dzeros = False) -> np.ndarray:

    '''Compute Intersection Over Union (IoU) between two sets of bounding boxes - 
    FINISH LATER.
    '''

    area, area_inter, area_union = boxes_stats(boxes_true)
    areas_ratio = area_inter / area_union
    if dzeros:
        areas_ratio[np.diag_indices_from(areas_ratio)] = 0

    return areas_ratio


def boxes_inclusion(boxes_true: np.ndarray, dzeros = False) -> np.ndarray:

    '''Compute Asymmetric Intersection over Union (IoU) of two sets of bounding boxes
    '''

    area, area_inter, area_union = boxes_stats(boxes_true)
    areas_ratio = area_inter / area
    if dzeros:
        areas_ratio[np.diag_indices_from(areas_ratio)] = 0

    return areas_ratio

def extract_dict_from_json_codeblock(text: str) -> dict:
    '''
    Fill later'''

    if text is None:
        raise ValueError("No text provided")
    
    # 1 Pull out the JSON block if fenced; otherwise use the whole text
    m = re.search(
        r'```(?:json)?\s*([\s\S]*?)\s*```', text, flags=re.IGNORECASE)
    s = (m.group(1) if m else text).strip()

    #2 Unwrap exactly one extra pair of outer braces if they exist
    if s.startswith('{{}') and s.endswith('}}'):
        s = s[1:-1].strip()

    #3 If not starting/ending with braces, pull out the first {...} chunk (simple)
    if not (s.startswith('{') or not s.endswith('}')):
        m2 = re.search(r"\{[\s\S]*?\}", s)
        if not m2:
            raise ValueError("No JSON object found in the provided text.")
        s = m2.group(0).strip()

    #4 Try strict JSON first
    try:
        data = json.loads(s)
    except Exception:
        #5 Fallback: normalize JSON literals to Python and use ast.literal_eval
        py_block = (
            s.replace('null', 'None')
            .replace('true', 'True')
            .replace('false', 'False')
        )
        try:
            data = ast.literal_eval(py_block)
        except Exception as e:
            raise ValueError(f"Failed to parse as JSON or Python literal: {e}") 
        
    if not isinstance(data, dict):
        raise ValueError(f"Parsed object is not a dictionary, (got {type(data)})")
    
    return data

def get_optimal_worker_count(ram_per_worker_gb=1.5, system_reserve_gb=4.0):
    """
    Calculates safe worker count based on Available RAM vs CPU Cores.
    """
    # 1. Get Total System RAM in GB
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    # 2. Calculate "Safe" RAM (Total - OS Overhead)
    available_ram_gb = total_ram_gb - system_reserve_gb
    
    # 3. Estimate how many workers fit in that RAM
    ram_based_limit = int(available_ram_gb / ram_per_worker_gb)
    
    # 4. Get CPU Core count
    cpu_count = os.cpu_count() or 1
    
    # 5. The Verdict: Take the smaller number (Bottleneck is usually RAM)
    # Ensure at least 1 worker, but don't exceed CPU count
    optimal_workers = max(1, min(ram_based_limit, cpu_count))
    
    print(f"--- [RESOURCE MANAGER] ---")
    print(f"Total RAM: {total_ram_gb:.1f} GB")
    print(f"Safe RAM Available: {available_ram_gb:.1f} GB")
    print(f"CPU Cores: {cpu_count}")
    print(f"Calculated Max Workers: {optimal_workers} (based on {ram_per_worker_gb}GB/worker)")
    print(f"--------------------------")
    
    return optimal_workers

def cleanup_resource(path: Path, force_cleanup: bool = False):
    """
    Safely deletes a directory or file if force_cleanup is True.
    Useful for 'clean as you go' to save disk space.
    """
    if not force_cleanup:
        logging.debug(f"🛑 Skipping cleanup for: {path} (Cleanup disabled)")
        return

    if not path.exists():
        logging.warning(f"⚠️  Cleanup requested but path not found: {path}")
        return

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        logging.info(f"♻️  Garbage Collected: {path.name}")
    except Exception as e:
        logging.error(f"❌ Failed to cleanup {path}: {e}")