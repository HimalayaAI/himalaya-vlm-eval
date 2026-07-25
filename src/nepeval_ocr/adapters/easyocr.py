"""EasyOCR adapter for local inference.

Requires easyocr:
    pip install -e '.[easyocr]'

Usage:
    from nepeval_ocr.adapters.easyocr import EasyOCRAdapter
    adapter = EasyOCRAdapter()
    text = adapter.evaluate_sample(image)
"""
from typing import Any, Dict, Tuple
from PIL.Image import Image
from .base import BaseOCRAdapter

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class EasyOCRAdapter(BaseOCRAdapter):
    """Adapter for EasyOCR model.
    
    Uses EasyOCR's default models with Nepali language support.
    Downloads pre-trained models on first run.
    """
    
    def __init__(self, langs: list = ["ne", "en"], device: str = None):
        """
        Initialize EasyOCR adapter.
        
        Args:
            langs: List of languages to use. Default is ["ne", "en"] for Nepali + English.
            device: Device to run on. Options: "auto", "cpu", "cuda".
                    If None, auto-detects GPU availability.
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR dependencies not installed. Install with: "
                "pip install -e '.[easyocr]'"
            )
        
        # Determine device
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.reader = easyocr.Reader(lang_list=langs, gpu=(self.device == "cuda"))

    def evaluate_sample(self, image: Image) -> str:
        """Run OCR on a single image and return transcribed text."""
        text, _ = self.evaluate_sample_with_stats(image)
        return text
    
    def evaluate_sample_with_stats(self, image: Image) -> Tuple[str, Dict[str, Any]]:
        """
        Run OCR on a single image and return text with timing stats.
        
        Returns:
            Tuple of (text, stats_dict)
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert image to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Convert to numpy array for EasyOCR
            import numpy as np
            img_np = np.array(image)
            
            # Run OCR
            results = self.reader.readtext(img_np)
            
            # Extract text from results (format: [bbox, text, confidence])
            text = " ".join([res[1] for res in results]).strip()
            
            latency = time.perf_counter() - start_time
            
            stats = {
                "latency_sec": latency,
                "device": self.device,
                "text_length": len(text),
                "detections": len(results),
            }
            
            return text, stats
            
        except Exception as e:
            latency = time.perf_counter() - start_time
            return "", {
                "latency_sec": latency,
                "error": str(e),
            }
