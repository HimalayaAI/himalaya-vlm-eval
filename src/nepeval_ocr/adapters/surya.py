"""Surya OCR adapter for local inference.

Requires surya-ocr to be installed: `pip install surya-ocr`

Surya auto-spawns an inference server (vllm for GPU or llama.cpp for CPU).
For CPU-only inference, install llama.cpp and set up the server manually,
or let surya spawn it automatically on first use.

Usage:
    from nepeval_ocr.adapters.surya import SuryaAdapter
    adapter = SuryaAdapter()
    text = adapter.evaluate_sample(image)
"""
import os
from typing import Any, Dict, Tuple
from PIL.Image import Image
from .base import BaseOCRAdapter

try:
    import surya
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor
    SURYA_AVAILABLE = True
except ImportError:
    SURYA_AVAILABLE = False


class SuryaAdapter(BaseOCRAdapter):
    """Adapter for Surya OCR engine.
    
    Uses the surya-ocr package which auto-spawns vllm (GPU) or llama.cpp (CPU)
    inference server on first use.
    
    For CPU-only machines, install llama.cpp server and point to it:
        export SURYA_INFERENCE_URL=http://localhost:8000/v1
        llama.cpp -m <model> -c 8192 --port 8000
    """
    
    def __init__(self, language: str = "ne"):
        """
        Initialize Surya adapter.
        
        Args:
            language: Language code. Surya supports multiple languages;
                      use "ne" for Nepali.
        """
        if not SURYA_AVAILABLE:
            raise ImportError(
                "Surya not installed. Install with: pip install surya-ocr"
            )
        
        self.lang = language
        self._predictor = None
        self._manager = None
        
        # Pre-validate backend availability by trying to create manager
        try:
            # Set llamacpp backend for CPU inference
            import surya.settings
            if not surya.settings.settings.SURYA_INFERENCE_BACKEND:
                surya.settings.settings.SURYA_INFERENCE_BACKEND = "llamacpp"
            
            # Try to create a manager to validate backend is available
            from surya.inference import SuryaInferenceManager
            test_manager = SuryaInferenceManager()
        except Exception as e:
            raise RuntimeError(
                f"Surya backend initialization failed: {e}. "
                "For CPU inference, install llama.cpp: "
                "macOS: brew install llama.cpp | Linux: download from "
                "https://github.com/ggml-org/llama.cpp/releases"
            )
    
    def _init_predictor(self):
        """Initialize Surya predictor and inference manager."""
        if self._predictor is not None:
            return
        
        self._manager = SuryaInferenceManager()
        
        # Configure for Nepali language if needed
        # Surya v2 uses multilingual models by default
        self._predictor = RecognitionPredictor(self._manager)
    
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
        self._init_predictor()
        
        import time
        start_time = time.perf_counter()
        
        try:
            # Surya returns blocks with text. For single images,
            # we typically want the first block's text.
            # The output format: list of pages, each with blocks
            predictions = self._predictor([image])
            
            if not predictions or not predictions.pages:
                return "", {"latency_sec": time.perf_counter() - start_time}
            
            # Extract text from first page's first block
            page = predictions.pages[0]
            blocks = page.blocks
            
            if not blocks:
                return "", {"latency_sec": time.perf_counter() - start_time}
            
            # Combine text from all blocks (sometimes text is split)
            full_text = "\n".join(b.text.strip() for b in blocks if b.text)
            
            latency = time.perf_counter() - start_time
            
            stats = {
                "latency_sec": latency,
                "block_count": len(blocks),
                "text_length": len(full_text),
            }
            
            return full_text, stats
            
        except RuntimeError:
            raise
        except Exception as e:
            latency = time.perf_counter() - start_time
            return "", {
                "latency_sec": latency,
                "error": str(e),
            }
