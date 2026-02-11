import shutil
import re
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from ContentMasker import ContentMasker
from MarkdownEnricher import MarkdownEnricher
from OCR import OCR
import img2pdf

class PageProcessor:
    def __init__(self, base_path: str, max_workers: int = 5):
        self.base_path = Path(base_path).resolve()
        self.max_workers = max_workers # <--- The Safety Valve
        
        self.masker = ContentMasker()
        try:
            print(">>> Attempting to load Primary Model (Qwen)...")
            self.ocr_engine = OCR("Qwen/Qwen3-VL-235B-A22B-Instruct")

        except Exception as e:
            logging.warning(f"⚠️ Primary model failed to load. Error: {e}")
            print(">>> Switching to Fallback Model (Llama)...")
            self.ocr_engine = OCR("meta-llama/llama-4-maverick-17b-128e-instruct")
        # Pass the worker count to the enricher too
        self.enricher = MarkdownEnricher(self.base_path, self.ocr_engine, max_workers)

    def process_and_mask(self):
        """Creates masked images (white boxes) for the main OCR pass."""
        input_dir = self.base_path / "pages"
        output_dir = self.base_path / "processed_pages"
        ignore_dir = self.base_path / "ignore_bounding_box"

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # Numerical sorting preserved!
        page_files = sorted(
            input_dir.glob("page_*.jpg"),
            key=lambda x: int(re.search(r"page_(\d+)", x.name).group(1))
        )
        
        if not page_files:
            print(f"[WARN] No images found in {input_dir}")
            return

        print(f"Found {len(page_files)} pages to mask.")

        for img_path in page_files:
            match = re.search(r"page_(\d+)", img_path.name)
            if not match: continue
            
            page_num = int(match.group(1))
            meta_path = ignore_dir / f"page_{page_num}" / "non_text_pairs.json"
            save_path = output_dir / img_path.name

            masked_image = self.masker.process_page(img_path, meta_path, page_num)
            masked_image.save(save_path, "PNG")

    def create_intermediate_pdf(self) -> Optional[Path]:
        img_dir = self.base_path / "processed_pages"
        pdf_name = f"{self.base_path.name}_masked.pdf"
        output_pdf = self.base_path / pdf_name
        
        images = sorted(
            [str(p) for p in img_dir.glob("*.jpg")],
            key=lambda x: int(re.search(r"page_(\d+)", x).group(1))
        )
        
        if not images:
            return None

        with open(output_pdf, "wb") as f:
            f.write(img2pdf.convert(images))
        return output_pdf

    def generate_final_markdown(self) -> Path:
        """Runs Main OCR (PARALLEL), Enriches content (PARALLEL), and saves."""
        
        processed_dir = self.base_path / "processed_pages"
        image_files = sorted(
            processed_dir.glob("*.jpg"),
            key=lambda x: int(re.search(r"page_(\d+)", x.name).group(1))
        )
        
        print(f"\n--- Starting Main OCR Pass (Workers: {self.max_workers}) ---")
        
        # Helper for the thread pool
        def process_page(path):
            print(f"  > Sending {path.name}...")
            try:
                return self.ocr_engine.inference_model("markdown", str(path))
            except Exception as e:
                print(f"  ⚠️ Primary model failed after retries. Switching to Backup...")
                try:
                    self.ocr_engine = OCR("meta-llama/llama-4-maverick-17b-128e-instruct")
                    return self.ocr_engine.inference_model("markdown", str(path))
                except Exception as e_backup:
                    error_msg = f"> **[OCR Failed] Both Primary and Backup models failed. Final Error: {str(e_backup)}**"
                    print(f"  ❌ {error_msg}")
                    return error_msg


        # Parallel Execution
        # executor.map preserves the order of the input list!
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_page, image_files))

        full_content = "\n\n---\n\n".join(results)
        
        # Enrichment Pass (Uses the max_workers passed in __init__)
        final_content = self.enricher.enrich(full_content)
        final_content = final_content.replace("```", "")
        
        original_filename = self.base_path.name.replace("_dla", "")
        final_md_filename = f"{original_filename}.md"
        # final_md_filename = Path(original_filename).with_suffix(".md")

        output_md_path = self.base_path/final_md_filename
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(final_content)
            
        print(f"\nSUCCESS: Final markdown saved to {output_md_path}")
        return output_md_path