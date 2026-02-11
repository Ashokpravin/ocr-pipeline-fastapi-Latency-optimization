import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class ContentMasker:
    """
    Responsible for visual pre-processing:
    1. Reads exclusion coordinates (tables/figures).
    2. Masks those regions with white space.
    3. Writes placeholder text (e.g., "PAGE 1 figure_0: HERE") for the OCR to read.
    """

    def __init__(self, font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size: int = 25):
        self.font_size = font_size
        self.font_path = font_path
        self._load_font()

    def _load_font(self):
        try:
            self.font = ImageFont.truetype(self.font_path, size=self.font_size)
        except OSError:
            print(f"[WARN] Font not found at {self.font_path}. Using default.")
            self.font = ImageFont.load_default()

    def process_page(self, image_path: Path, metadata_path: Path, page_num: int) -> Image.Image:
        """
        Loads an image, masks coordinates found in metadata_path, and draws placeholders.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        
        if not metadata_path.exists():
            # If no metadata (no tables/figures), return original image
            return image

        captions, coordinates = self._get_coordinates(metadata_path)
        return self._apply_masks(image, captions, coordinates, page_num)

    def _get_coordinates(self, json_path: Path) -> Tuple[List[str], List[List[int]]]:
        """Parses the DLA exclusion JSON."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Expecting data format: [{"object": "table", "bbox": [x1, y1, x2, y2]}, ...]
        captions = [item["object"] for item in data]
        coordinates = [item["bbox"] for item in data]
        return captions, coordinates

    def _apply_masks(self, image: Image.Image, captions: List[str], coordinates: List[List[int]], page_num: int) -> Image.Image:
        """Draws white rectangles and placeholder text."""
        draw = ImageDraw.Draw(image)
        
        fig_count = 0
        table_count = 0

        for caption, (x_min, y_min, x_max, y_max) in zip(captions, coordinates):
            # 1. Draw White Mask
            draw.rectangle([x_min, y_min, x_max, y_max], fill="white", outline=None)

            # 2. Generate Placeholder Tag
            tag = ""
            lower_cap = caption.lower()
            
            if "figure" in lower_cap:
                tag = f"PAGE {page_num} figure_{fig_count}: HERE"
                fig_count += 1
            elif "table" in lower_cap:
                tag = f"PAGE {page_num} table_{table_count}: HERE"
                table_count += 1
            else:
                # Fallback for other objects
                tag = f"PAGE {page_num} {caption}: HERE"

            # 3. Write Tag in the center of the white box (or top left)
            # Drawing at (x_min, y_min) is safest
            draw.text((x_min, y_min), tag, font=self.font, fill="black")
            
        return image