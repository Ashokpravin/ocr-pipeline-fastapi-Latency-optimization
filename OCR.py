import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from PIL import Image
import math
import io
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
import sys

# Ensure config is available 
try:
    import config
except ImportError:
    print("[WARN] 'config.py' not found. Ensure system prompts are defined.")
    config = None

#Simple Logger configuration
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
logger = logging.getLogger(__name__)

class OCR:
    def __init__(self, model_name: str):
        load_dotenv()
        self.model_name = model_name
        self._configure_model()

    def _configure_model(self):
        """Maps model names to environment variables."""
        if self.model_name == "meta-llama/llama-4-maverick-17b-128e-instruct":
            self.api_key = os.getenv("MODEL_API_KEY_LLAMA")
            self.model_url = os.getenv("MODEL_URL_PATH_LLAMA")
            self.MAX_PIXELS = 33177600
        elif self.model_name == "Qwen/Qwen3-VL-235B-A22B-Instruct":
            self.api_key = os.getenv("MODEL_API_KEY_QWEN_3")
            self.model_url = os.getenv("MODEL_URL_PATH_QWEN_3")
            self.MAX_PIXELS = 66355200
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        if not self.api_key or not self.model_url:
            print(f"[WARN] API credentials missing for {self.model_name}")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _process_image(self, image_path):
        """         
        Reads image, resizes if too big, and returns base64 string.         
        """
        with Image.open(image_path) as img:             
            width, height = img.size            
            total_pixels = width * height            

            # Check if compression is needed
            if total_pixels > self.MAX_PIXELS:                 
                # Calculate scaling factor to get under the limit
                # scale = sqrt(target / current)                
                scale_factor = math.sqrt(self.MAX_PIXELS / total_pixels)    

                # Apply a tiny safety margin (0.99) to ensure we stay under                
                new_width = int(width * scale_factor * 0.99)                 
                new_height = int(height * scale_factor * 0.99)                                 
                
                print(f"[OCR] Compressing {os.path.basename(image_path)}: {width}x{height} -> {new_width}x{new_height}")                                 
                
                # High-quality downsampling               
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)             
                
            # Convert to RGB (in case of PNG with alpha channel)
            if img.mode in ("RGBA", "P"): 
                img = img.convert("RGB") 
                
            # Save to memory buffer instead of disk 
            buffer = io.BytesIO() 
            img.save(buffer, format="JPEG", quality=90) # Standard high quality JPEG
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _get_system_prompt(self, task_type: str) -> str:
        """Selects the correct prompt based on the object type (figure/table/text)."""
        if not config:
            return "OCR this image in detail."

        task_type = task_type.lower()
        if "figure" in task_type:
            return config.SYSTEM_PROMPT_FIGURE
        elif "table" in task_type:
            return config.SYSTEM_PROMPT_TABLE
        
        return config.SYSTEM_PROMPT_MARKDOWN

    @retry(
        # Stop after 3 attempts (1 original + 2 retries)
        stop= stop_after_attempt(3),

        # Wait 2^x * 1 seconds between retries (2s, 4s, 8s...)
        wait= wait_exponential(multiplier=1, min=2, max=10),

        # Retry ONLY on network errors or server 5xx errors
        retry= retry_if_exception_type(requests.exceptions.RequestException),

        # Log a warning before waiting to retry
        before_sleep= before_sleep_log(logger, logging.WARNING)
    )

    def inference_model(self, task_type: str, img_path: str) -> str:
        """
        Main entry point. Encodes image, selects prompt, calls API, returns text.
        Always returns a String (never a Dict/None) to be safe for Markdown writing.
        """

        b64_image = self._process_image(img_path)
        sys_prompt = self._get_system_prompt(task_type)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "OCR this scanned image in full details."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]}
            ]
        }

        response = requests.post(self.model_url, json=payload, headers=self.headers, timeout=300)
        response.raise_for_status()
        # Process Success (This part is only reached if status is 200 OK)
        try:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                return "> **[OCR Warning] Model returned empty response.**"
            return content
        except requests.exceptions.JSONDecodeError:
            return f"> **[OCR Error] Invalid JSON response.** (Status {response.status_code})"
            