"""PaddleOCR adapter for local inference.

Requires paddleocr and paddlepaddle:
    pip install -e '.[paddle]'
    pip install paddlepaddle  # from https://www.paddlepaddle.org.cn/en/install/quick

Usage:
    from nepeval_ocr.adapters.paddle import PaddleOCRAdapter
    adapter = PaddleOCRAdapter()
    text = adapter.evaluate_sample(image)
"""
from typing import Any, Dict, Tuple
from PIL.Image import Image
from .base import BaseOCRAdapter

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


class PaddleOCRAdapter(BaseOCRAdapter):
    """Adapter for PaddleOCR model.
    
    Uses PaddleOCR with Devanagari script support for Nepali.
    Downloads pre-trained models on first run.
    
    Note: PaddleOCR 3.x requires paddlepaddle installation which is not available
    on PyPI. Install from: https://www.paddlepaddle.org.cn/en/install/quick
    """
    
    def __init__(self, lang: str = "ne", device: str = None):
        """
        Initialize PaddleOCR adapter.
        
        Args:
            lang: Language code to use. Default is "ne" (Nepali) which uses Devanagari script.
                  Other Devanagari options: hi (Hindi), mr (Marathi), sa (Sanskrit).
            device: Device to run on. Options: "auto", "cpu", "cuda".
                    If None, auto-detects GPU availability.
                    Note: PaddleOCR 3.x ignores this and uses CPU by default.
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR dependencies not installed. Install with: "
                "pip install -e '.[paddle]' and pip install paddlepaddle"
            )
        
        # PaddleOCR 3.x uses new API without explicit device parameter
        # Device selection is handled internally by PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            ocr_version="PP-OCRv5",
            show_log=False
        )
        self.device = device or "cpu"

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
            
            # Convert to numpy array for PaddleOCR
            import numpy as np
            img_np = np.array(image)
            
            # Run OCR
            # Note: PaddleOCR 3.x uses different result format
            results = self.ocr.ocr(img_np, cls=True)
            
            # Extract text from results
            # Format varies between versions, handle both
            if not results:
                text = ""
            elif isinstance(results, list) and len(results) > 0:
                # New format: [[(bbox, (text, conf)), ...], ...]
                text_parts = []
                for page in results:
                    if isinstance(page, list):
                        for item in page:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                text_parts.append(item[1][0])
                text = " ".join(text_parts).strip()
            else:
                text = ""
            
            latency = time.perf_counter() - start_time
            
            stats = {
                "latency_sec": latency,
                "device": self.device,
                "text_length": len(text),
            }
            
            return text, stats
            
        except Exception as e:
            latency = time.perf_counter() - start_time
            return "", {
                "latency_sec": latency,
                "error": str(e),
            }
