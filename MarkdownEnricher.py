import re
from pathlib import Path
from OCR import OCR
from concurrent.futures import ThreadPoolExecutor, as_completed

class MarkdownEnricher:
    """
    Responsible for post-processing:
    1. Parses the rough Markdown from the main OCR pass.
    2. Finds placeholders (e.g., "PAGE 1 figure_0: HERE").
    3. Locates the corresponding cropped image from DLA.
    4. Runs specialized OCR on that crop.
    5. Injects the result back into the Markdown.
    """

    def __init__(self, base_path: Path, ocr_engine: OCR, max_workers: int = 1):
        self.base_path = base_path
        self.cropped_dir = base_path / "cropped_objects"
        self.ocr = ocr_engine
        self.max_workers = max_workers
        
        # Regex to find tags like "PAGE 5 figure_2: HERE"
        # Matches: Group 1 (Page #), Group 2 (Type), Group 3 (Index)
        self.placeholder_pattern = re.compile(
            r"page\s+(\d+)\s+(figure|table)(?:[^\d\n]*(\d+))?.*?here",
            re.IGNORECASE
        )

    def enrich(self, markdown_content: str) -> str:
        print(f"\n--- Enriching Markdown (Workers: {self.max_workers}) ---")
        
        # 1. Identify all unique placeholders first
        matches = list(self.placeholder_pattern.finditer(markdown_content))
        
        if not matches:
            print("No figures or tables found to enrich.")
            return markdown_content

        print(f"Found {len(matches)} items to process.")
        
        # 2. Run OCR (Parallel or Sequential based on max_workers)
        # We store results in a dictionary: { "PAGE 1 figure_0: HERE": "OCR TEXT..." }
        replacements = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_match = {
                executor.submit(self._process_single_match, m): m.group(0) 
                for m in matches
            }
            
            # Gather results as they finish
            for future in as_completed(future_to_match):
                original_tag = future_to_match[future]
                try:
                    ocr_result = future.result()
                    replacements[original_tag] = ocr_result
                except Exception as e:
                    print(f"Error processing {original_tag}: {e}")
                    replacements[original_tag] = f"> **Error: {e}**"

        # 3. Perform the replacement in the text
        # The callback just looks up the pre-calculated result
        def replacement_callback(match):
            return replacements.get(match.group(0), match.group(0))

        return self.placeholder_pattern.sub(replacement_callback, markdown_content)

    def _process_single_match(self, match) -> str:
        """Worker function: Resolves path and runs OCR."""
        page_num = match.group(1)
        obj_type = match.group(2).lower()
        obj_index = match.group(3)
        
        # Construct path
        page_folder_name = f"page_{page_num}.jpg"
        target_crop = (
            self.cropped_dir / 
            page_folder_name / 
            obj_type / 
            f"{obj_type}_{obj_index}.png"
        )

        # Fallback for folder naming (handling plural/singular issues)
        if not target_crop.exists():
             target_crop = (
                self.cropped_dir / 
                page_folder_name / 
                obj_type.rstrip('s') / 
                f"{obj_type}_{obj_index}.png"
            )

        if not target_crop.exists():
            return f"\n> **[Missing Crop] Could not find image for {obj_type} {obj_index}**\n"

        print(f"    - Processing {target_crop.name}...")
        try: 
            return f"\n{self.ocr.inference_model(obj_type, str(target_crop))}\n"
        
        except Exception as e:
            print(f"  ⚠️ Primary model failed for {obj_type}. Switching to Backup...")

            try:
                self.ocr = OCR("meta-llama/llama-4-maverick-17b-128e-instruct")
                return f"\n{self.ocr.inference_model(obj_type, str(target_crop))}\n"
            
            except Exception as e_backup:
                error_msg = f"\n> **[OCR Failed] Both Primary and Backup models failed. Final Error: {str(e_backup)}**\n"
                print(f"  ❌ {error_msg}")
                return error_msg
