import sys
import shutil
import json
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from collections import defaultdict
import copy

import numpy as np
import cv2
import supervision as sv
from paddleocr import LayoutDetection

# Assumes utils.py contains boxes_inclusion and boxes_iou
from utils import *

class DLA:
    """
    Document Layout Analysis Pipeline.
    Handles Layout Analysis (PaddleOCR), heuristic merging of regions,
    and output generation.
    """

    def __init__(self):
        self.package_dir = Path(__file__).resolve().parent.parent
        self._load_config()
        self._init_palette()
        self._init_paddle_model()
        self._reset_state()

    def _load_config(self):
        json_file = self.package_dir / "resources" / "dla.vars.json"
        self.dla_vars = {}
        if json_file.exists():
            with open(json_file, 'r') as f:
                self.dla_vars = json.load(f)

    def _init_palette(self):
        """Define custom colors for visualization."""
        self.class_colors = [
            sv.Color(255, 0, 0),       # Red: Caption
            sv.Color(0, 255, 0),       # Green: Footnote
            sv.Color(0, 0, 255),       # Blue: Formula
            sv.Color(255, 255, 0),     # Yellow: List-item
            sv.Color(255, 0, 255),     # Magenta: Page-footer
            sv.Color(0, 255, 255),     # Cyan: Page-header
            sv.Color(128, 0, 128),     # Purple: Picture
            sv.Color(128, 128, 0),     # Olive: Section-header
            sv.Color(128, 128, 128),   # Gray: Table
            sv.Color(0, 128, 128),     # Teal: Text
            sv.Color(128, 0, 0),       # Maroon: Title
        ]

    def _init_paddle_model(self, model_name: str = "PP-DocLayout_plus-L"):
        """Initializes the PaddleOCR LayoutDetection model and label mappings."""
        self.psession = LayoutDetection(model_name=model_name)

        # Map raw Paddle labels to our internal pipeline labels
        self.map_labels = {
            "paragraph_title": "text", 
            "image": "figure", 
            "text": "text",
            "number": "text", 
            "abstract": "text", 
            "content": "text",
            "figure_title": "text", 
            "formula": "formula", 
            "table": "table",
            "reference": "text", 
            "doc_title": "text", 
            "footnote": "text",
            "header": "text", 
            "algorithm": "figure", 
            "footer": "abandon",
            "seal": "figure", 
            "chart": "figure", 
            "formula_number": "text",
            "aside_text": "abandon", 
            "reference_content": "text",
        }

        # Create mapping indices for efficient numpy operations
        raw_classes = list(self.map_labels.keys())
        mapped_classes = sorted(list(set(self.map_labels.values())))
        
        # Map raw index to consolidated index
        self.ind_map = np.array([mapped_classes.index(self.map_labels[k]) for k in raw_classes])
        
        # Dict to lookup string name from int index
        self.model_class_names = dict(enumerate(mapped_classes))

    def _reset_state(self, out_dir: Optional[str] = None):
        """Resets the processing state variables."""
        self.files_list: List[str] = []
        self.images: List[np.ndarray] = []
        self.results: List[sv.Detections] = []
        self.annotated_images: List[np.ndarray] = []
        self.cropped_objects: List[Dict] = []
        self.out_dir = Path(out_dir) if out_dir else Path("./output/")

    def _get_dir_paths(self) -> Dict[str, Path]:
        return {
            "pages": self.out_dir / "pages",
            "cropped": self.out_dir / "cropped_objects",
            "labeled": self.out_dir / "labeled",
        }

    # =========================================================================
    # Entry Point
    # =========================================================================

    def set_images(self, image_paths: List[str], output_dir: Path):
        """
        Loads the prepared images into the pipeline state.
        This replaces the old 'upload' and file conversion logic.
        """
        self._reset_state(str(output_dir))
        self.files_list = image_paths
        
        # Load images into memory for processing
        self.images = []
        for p in self.files_list:
            img = cv2.imread(p)
            if img is None:
                print(f"[WARN] Failed to load image: {p}")
                continue
            self.images.append(img)

        if not self.images:
            raise ValueError("No valid images loaded in DLA pipeline.")
        

    # =========================================================================
    # CORE ANALYSIS & MERGING LOGIC
    # =========================================================================

    def analyze(self, conf: float = 0.38, iou: float = 0.5, filter_dup: bool = True, merge_visual: bool = True):
        """
        Main pipeline execution: Inference -> Duplication Filter -> Heuristic Merging -> Cropping
        """
        self._clean_previous_results()

        if not self.images:
            return

        for ii,img in enumerate(self.images):
            # 1. Inference
            raw_output = self.psession.predict(img, layout_nms=False, threshold=conf)[0]
            detections = self._convert_pp_to_sv(raw_output, img.shape)


            # 2. Filter Duplicates & Merge logic
            if filter_dup:
                # Merge overlapping text
                detections = self._merge_object_pair(
                    detections, "text", tlabel="abandon", threshold=iou
                )
                # Merge text inside/near tables/figures -> label as formula
                detections = self._merge_object_pair(
                    detections, ["text", "table", "figure"], tlabel="formula", threshold=iou
                )
                # Merge 'abandon' objects into valid objects
                detections = self._merge_object_pair(
                    detections, "abandon", 
                    tlabel=["figure", "table", "formula", "text", "abandon"], 
                    threshold=iou
                )
                # Clean up overlaps among main classes
                detections = self._merge_object_pair(
                    detections, ["text", "figure", "table", "formula"]
                )

            # 3. Geometric Merging (Captions, etc.)
            if merge_visual:
                detections = self._merge_formula_text(detections)
                detections = self._merge_text_figure_table(detections)

            self.results.append(detections)
            self.cropped_objects.append(self._crop_objects(img, detections))

    def _clean_previous_results(self):
        self.results = []
        self.annotated_images = []
        self.cropped_objects = []
        dirs = self._get_dir_paths()
        if dirs["cropped"].exists(): shutil.rmtree(dirs["cropped"])
        if dirs["labeled"].exists(): shutil.rmtree(dirs["labeled"])

    # =========================================================================
    # MERGING HELPERS
    # =========================================================================

    def _merge_object_pair(self, res: sv.Detections, rlabel: Union[str, List], 
                           tlabel: Union[str, List] = None, threshold: float = 0.0) -> sv.Detections:
        """
        Generic merge function: checks for inclusion/intersection between class 'rlabel'
        and class 'tlabel'. If conditions met, unions the boxes.
        """
        merged_res = copy.deepcopy(res)
        if isinstance(rlabel, str): rlabel = [rlabel]
        if isinstance(tlabel, str): tlabel = [tlabel]

        has_changes = True
        while has_changes:
            has_changes = False
            
            if not merged_res.class_id.size:
                return merged_res

            # Filter masks
            current_classes = merged_res.data['class_name']
            is_target_class = np.array([c in rlabel for c in current_classes])

            if not np.any(is_target_class):
                return merged_res

            # Calculate inclusion matrix
            mat_dist = boxes_inclusion(merged_res.xyxy, dzeros=True)
            keep_mask = np.ones(len(merged_res.class_id), dtype=bool)

            for i in range(len(merged_res.class_id)):
                if is_target_class[i]:
                    # Determine valid merge candidates based on labels
                    if tlabel is None:
                        # Merge with same class
                        is_candidate_class = (current_classes == current_classes[i])
                    else:
                        # Merge with specific target classes
                        is_candidate_class = np.array([c in tlabel for c in current_classes])

                    # Get inclusion scores for this box against candidates
                    scores = mat_dist[i, :].copy()
                    scores[~is_candidate_class] = 0.0
                    scores[scores < threshold] = 0.0

                    # If valid intersections found
                    if np.sum(scores) > 0:
                        merge_indices = np.nonzero(scores)[0]
                        merged_res = self._union_objects(merged_res, i, merge_indices)
                        
                        # Mark merged objects for removal
                        keep_mask[merge_indices] = False
                        
                        # Zero out processed relationships to prevent double counting in this loop
                        mat_dist[merge_indices, :] = 0
                        mat_dist[:, merge_indices] = 0
                        has_changes = True

            merged_res = self._remove_objects(merged_res, keep_mask)

        return merged_res

    def _merge_text_figure_table(self, res: sv.Detections) -> sv.Detections:
        """
        Heuristic: Merges text captions located at the bottom of figures/tables,
        or titles at the top of tables.
        """
        detections = copy.deepcopy(res)
        if not detections.class_id.size:
            return detections

        names = detections.data["class_name"]
        is_text = names == "text"
        is_fig = names == "figure"
        is_table = names == "table"
        
        keep_mask = np.ones(len(names), dtype=bool)

        if not (np.sum(is_text) * (np.sum(is_fig) + np.sum(is_table))):
            return detections

        # ---------------------------------------------------------
        # 1. Merge Text at BOTTOM of Figures/Tables (Captions)
        # ---------------------------------------------------------
        for i in range(len(names)):
            if keep_mask[i] and (is_fig[i] or is_table[i]):
                
                bbox = detections.xyxy
                
                # Geometric filters
                # Box is below current object (y_min > current y_min)
                is_below = bbox[:, 1] > bbox[i, 1] 
                
                # Vertical projection intersection (IOU Vert > 0)
                iou_vert = self._bbox_iou_vert(bbox)
                is_vert_aligned = iou_vert[i, :] > 0
                
                candidates = is_below * is_vert_aligned * (~is_text)
                
                # Constrain search to nearest neighbors vertically
                if np.sum(candidates):
                    is_below *= bbox[:, 1] < bbox[candidates, 1].min()

                heights = bbox[:, 3] - bbox[:, 1]
                widths = bbox[:, 2] - bbox[:, 0]
                
                # Distance between current bottom and candidate top
                dist_y = bbox[:, 1] - bbox[i, 3]
                
                # Heuristic: Distance should be less than height of candidate
                is_close_enough = (bbox[:, 3] - bbox[i, 3]) <= heights[i]

                valid_text_candidates = is_text * is_below * is_vert_aligned * is_close_enough

                if np.sum(valid_text_candidates) == 1:
                    idx = np.argmax(valid_text_candidates)
                    
                    should_merge = True
                    # Specific rules for Tables
                    if is_table[i] and dist_y[idx] > heights[idx]:
                        should_merge = False
                    
                    # Check for left shift (indentation logic)
                    if (bbox[idx, 0] < bbox[i, 0]) and (bbox[idx, 2] < bbox[i, 2]):
                        should_merge = False
                        
                    if is_fig[i]:
                         # Check geometric inclusion/distance
                         mat_inc = boxes_inclusion(bbox, dzeros=True)
                         center_x = (bbox[i, 2] + bbox[i, 0]) / 2
                         if (bbox[idx, 2] < center_x) and (mat_inc[i, idx] < 0.5):
                             should_merge = False

                    if should_merge:
                        detections = self._union_objects(detections, i, [idx])
                        is_text[idx] = False
                        keep_mask[idx] = False

                # Logic for multiple text lines (paragraph below figure)
                elif np.sum(valid_text_candidates) > 1:
                    indices = np.nonzero(valid_text_candidates)[0]
                    # Sort by Y position
                    indices = indices[np.argsort(bbox[indices, 1])]
                    
                    # Logic to stop merging if sequence breaks
                    # (This block contains the complex "stop_lp" logic from original code)
                    # Simplified for readability but logic preserved
                    candidates_to_merge = []
                    
                    is_main_included = (bbox[:, 0] >= bbox[i, 0]) & (bbox[:, 2] <= bbox[i, 2])
                    
                    # Check first candidate
                    if not is_main_included[indices[0]]:
                        pass # Continue checking logic below
                    
                    candidates_to_merge.append(indices[0])

                    for k in range(len(indices) - 1):
                        curr, next_box = indices[k], indices[k+1]
                        
                        # Stop conditions
                        dist_pair = bbox[next_box, 1] - bbox[curr, 3]
                        
                        # 1. Distance too large
                        if (2 * heights[curr] < dist_pair) or (2 * heights[next_box] < dist_pair): break
                        
                        # 2. Width mismatch significantly
                        if widths[i] / widths[curr] > 4: break

                        # 3. Not vertically aligned
                        if not iou_vert[curr, next_box]: break
                        
                        # 4. Alignment/Indentation shifts
                        if is_main_included[curr] and not is_main_included[next_box]: break
                        if not is_main_included[curr] and not is_main_included[next_box]: break
                        
                        # Left Shift checks
                        if not is_main_included[curr]:
                             if (bbox[next_box, 0] <= bbox[curr, 0]) or (bbox[next_box, 2] >= bbox[curr, 2]): break
                        
                        center_curr = (bbox[curr, 2] + bbox[curr, 0]) / 2
                        if bbox[next_box, 2] < center_curr: break

                        if (bbox[next_box, 0] < bbox[i, 0]) and (bbox[next_box, 2] < bbox[i, 2]): break

                        candidates_to_merge.append(next_box)

                    if candidates_to_merge:
                        detections = self._union_objects(detections, i, candidates_to_merge)
                        is_text[candidates_to_merge] = False
                        keep_mask[candidates_to_merge] = False

        # ---------------------------------------------------------
        # 2. Merge Text at TOP of Tables (Titles)
        # ---------------------------------------------------------
        for i in range(len(names)):
            if keep_mask[i] and is_table[i]:
                bbox = detections.xyxy
                
                # Check boxes ABOVE current table
                is_above = bbox[:, 3] < bbox[i, 3]
                is_vert_aligned = self._bbox_iou_vert(bbox)[i, :] > 0
                
                # Filter strictly to immediate neighbors above
                candidates = is_above * is_vert_aligned * (~is_text)
                if np.sum(candidates):
                    is_above *= bbox[:, 3] > bbox[candidates, 3].max()

                is_included = (bbox[:, 0] >= bbox[i, 0]) & (bbox[:, 2] <= bbox[i, 2])
                
                heights = bbox[:, 3] - bbox[:, 1]
                dist_y = bbox[i, 1] - bbox[:, 3]
                
                is_close = dist_y <= heights
                
                valid_candidates = is_text * is_above * is_vert_aligned * is_included * is_close

                if np.sum(valid_candidates):
                    # Ensure we don't merge things that are physically "inside" others (sanity check)
                    valid_candidates *= bbox[:, 3] > bbox[valid_candidates, 1].max()
                    
                    indices = np.nonzero(valid_candidates)[0]
                    detections = self._union_objects(detections, i, indices)
                    is_text[indices] = False
                    keep_mask[indices] = False

        return self._remove_objects(detections, keep_mask)

    def _merge_formula_text(self, res: sv.Detections) -> sv.Detections:
        """Merges fragmented text parts into formulas."""
        detections = copy.deepcopy(res)
        if not detections.class_id.size: return detections

        names = detections.data["class_name"]
        is_text = names == "text"
        is_formula = names == "formula"
        keep_mask = np.ones(len(names), dtype=bool)

        if not (np.sum(is_formula) * np.sum(is_text)):
            return detections

        # Strategy 1: Merge text inside formula boundaries (with aspect ratio check)
        for i in range(len(names)):
            if is_formula[i] and keep_mask[i]:
                bbox = detections.xyxy
                
                # Box inclusion heuristic
                is_inside = (bbox[i, 3] >= bbox[:, 3]) * (bbox[i, 1] <= bbox[:, 1])
                
                heights = bbox[:, 3] - bbox[:, 1]
                widths = bbox[:, 2] - bbox[:, 0]
                ratio_ok = np.maximum(heights, widths) / np.minimum(heights, widths) < 2
                
                candidates = keep_mask * is_text * is_inside * ratio_ok
                
                if np.sum(candidates):
                    indices = np.nonzero(candidates)[0]
                    detections = self._union_objects(detections, i, indices)
                    keep_mask[indices] = False

        # Strategy 2: Merge text below formula (vertical stack)
        for i in range(len(names)):
            if is_formula[i] and keep_mask[i]:
                bbox = detections.xyxy
                iou_vert = self._bbox_iou_vert(bbox)
                
                is_below = bbox[:, 1] > bbox[i, 1]
                is_aligned = iou_vert[i, :] > 0
                
                # Stop at next non-formula object
                blockers = is_below * is_aligned * (~is_formula)
                if np.sum(blockers):
                    is_below *= bbox[:, 1] < bbox[blockers, 1].min()
                
                candidates = keep_mask * is_text * is_below * is_aligned
                
                if np.sum(candidates):
                    indices = np.nonzero(candidates)[0]
                    detections = self._union_objects(detections, i, indices)
                    keep_mask[indices] = False

        return self._remove_objects(detections, keep_mask)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _bbox_iou_vert(self, bbox: np.ndarray) -> np.ndarray:
        """Calculates vertical intersection over union (ignoring X-axis)."""
        cbbox = bbox.copy()
        # Set x-coordinates to 0 and 1 effectively flattening width
        cbbox[:, [0, 2]] = [0, 1] 
        return boxes_iou(cbbox, dzeros=False)

    def _remove_objects(self, res: sv.Detections, mask: np.ndarray) -> sv.Detections:
        """Filters detections based on a boolean mask."""
        if not res.class_id.size: return res
        
        res.xyxy = res.xyxy[mask]
        res.confidence = res.confidence[mask]
        res.class_id = res.class_id[mask]
        res.data['class_name'] = res.data['class_name'][mask]
        return res

    def _union_objects(self, res: sv.Detections, base_idx: int, merge_indices: Union[List, np.ndarray]) -> sv.Detections:
        """Expands the bounding box at base_idx to include boxes at merge_indices."""
        indices = [base_idx] + list(merge_indices)
        vectors = res.xyxy[indices, :]
        
        # Calculate new bounding box (min x, min y, max x, max y)
        new_box = np.array([
            vectors[:, 0].min(), vectors[:, 1].min(),
            vectors[:, 2].max(), vectors[:, 3].max()
        ])
        res.xyxy[base_idx, :] = new_box
        return res

    def _convert_pp_to_sv(self, pp_result: dict, img_shape: Tuple) -> sv.Detections:
        """Converts PaddleOCR output format to Supervision Detections."""
        boxes = pp_result.get('boxes', [])
        if not boxes:
            return sv.Detections.empty()

        # Sort by score
        boxes.sort(key=lambda x: x["score"], reverse=True)

        xyxy = np.array([b["coordinate"] for b in boxes]).astype(int)
        conf = np.array([b["score"] for b in boxes])
        
        # Map class IDs
        raw_ids = np.array([b["cls_id"] for b in boxes]).astype(int)
        class_ids = self.ind_map[raw_ids]
        class_names = np.array([self.model_class_names[c] for c in class_ids])

        # Padding logic for visuals
        padding = min(img_shape[0], img_shape[1]) * 0.005
        for i, name in enumerate(class_names):
            if name in ["table", "formula", "figure"]:
                xyxy[i, :] = [
                    max(0, xyxy[i, 0] - padding),
                    max(0, xyxy[i, 1] - padding),
                    min(img_shape[1], xyxy[i, 2] + padding),
                    min(img_shape[0], xyxy[i, 3] + padding),
                ]

        return sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_ids,
            data={"class_name": class_names}
        )

    def _crop_objects(self, img: np.ndarray, detections: sv.Detections) -> Dict:
        """Crops detected objects from the original image."""
        if not detections.class_id.size:
            return {'objects': [], 'inc_mat': None}

        objects = []
        for i, (xyxy, name) in enumerate(zip(detections.xyxy, detections.data['class_name'])):
            crop = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])].copy()
            objects.append((name, crop))

        return {'objects': objects, 'inc_mat': boxes_inclusion(detections.xyxy)}

    # =========================================================================
    # VISUALIZATION & OUTPUT
    # =========================================================================

    def annotate_images(self):
        """Draws bounding boxes and labels on processed images."""
        self.annotated_images = []
        for img, det in zip(self.images, self.results):
            box_annotator = sv.BoxAnnotator(
                color=sv.ColorPalette(self.class_colors), thickness=3
            )
            label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette(self.class_colors), text_color=sv.Color(255, 255, 255)
            )
            
            ann_img = box_annotator.annotate(scene=img.copy(), detections=det)
            ann_img = label_annotator.annotate(scene=ann_img, detections=det)
            self.annotated_images.append(ann_img)

    def save_labeled_images(self) -> List[str]:
        if len(self.images) != len(self.annotated_images):
            self.annotate_images()

        dir_path = self._get_dir_paths()["labeled"]
        dir_path.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, (fname, img) in enumerate(zip(self.files_list, self.annotated_images)):
            save_path = dir_path / Path(fname).name
            cv2.imwrite(str(save_path), img)
            paths.append(str(save_path))
        return paths

    def save_cropped_objects(self) -> List[Dict]:
        dir_path = self._get_dir_paths()["cropped"]
        
        # Clean specific file directories if they exist
        for fname in self.files_list:
            page_dir = dir_path / Path(fname).name
            if page_dir.exists(): shutil.rmtree(page_dir)

        output_data = []
        for i, (fname, crop_data) in enumerate(zip(self.files_list, self.cropped_objects)):
            bname = Path(fname).name
            page_dir = dir_path / bname
            page_dir.mkdir(parents=True, exist_ok=True)

            file_record = defaultdict(list)
            file_record["file_path"] = fname
            
            counter = defaultdict(int)

            for class_name, img_crop in crop_data["objects"]:
                obj_dir = page_dir / class_name
                obj_dir.mkdir(exist_ok=True)
                
                img_name = f"{class_name}_{counter[class_name]}.png"
                img_path = obj_dir / img_name
                
                cv2.imwrite(str(img_path), img_crop)
                file_record[class_name].append(str(img_path))
                counter[class_name] += 1
            
            output_data.append(dict(file_record))
            
        return output_data
    
    def save_structure_json(self):
        """
        Saves bounding boxes for non-text objects (Tables, Figures) to JSON.
        Required for the Masking step in PageProcessor.
        """
        ignored_labels = {'text', 'formula', 'abandon'}
        
        # We need to save this inside "ignore_bounding_box/page_N"
        base_dir = self._get_dir_paths()["pages"].parent / "ignore_bounding_box"
        
        for i, result in enumerate(self.results):
            if not result.class_id.size: continue

            # Extract Data
            names = result.data["class_name"]
            boxes = result.xyxy
            
            non_text_data = []
            for name, box in zip(names, boxes):
                if name.lower() not in ignored_labels:
                    non_text_data.append({
                        "object": name,
                        "bbox": box.tolist()
                    })

            # Save to: /output/file_dla/ignore_bounding_box/page_N/non_text_pairs.json
            page_dir = base_dir / f"page_{i}"
            page_dir.mkdir(parents=True, exist_ok=True)
            
            json_path = page_dir / "non_text_pairs.json"
            with open(json_path, 'w', encoding="utf-8") as f:
                json.dump(non_text_data, f, indent=2)


    def run_vision_pipeline(self, image_paths: List[str], output_dir: Path, filter_dup=True, merge_visual=True):
        """One-shot function to run the whole process."""
        self.set_images(image_paths, output_dir)
        self.analyze(filter_dup=filter_dup, merge_visual=merge_visual)
        self.save_labeled_images()
        self.save_structure_json()
        return self.save_cropped_objects()