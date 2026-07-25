"""trOCR adapter for local inference.

Requires transformers and torch:
    pip install transformers torch

Usage:
    from nepeval_ocr.adapters.trocr import TrOCRAdapter
    adapter = TrOCRAdapter()
    text = adapter.evaluate_sample(image)
"""
import os
from typing import Any, Dict, Tuple
from PIL.Image import Image
from .base import BaseOCRAdapter

try:
    import torch
    from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class TrOCRAdapter(BaseOCRAdapter):
    """Adapter for trOCR (Transformer-based OCR) model.
    
    Uses the syubraj/TrOCR_Nepali model from Hugging Face.
    Loads the model into memory on initialization. For CPU-only inference
    with limited RAM, consider using torch.device("cpu") with appropriate
    batch size settings.
    
    The model uses VisionEncoderDecoder architecture (ViT encoder + decoder).
    """
    
    MODEL_NAME = "syubraj/TrOCR_Nepali"
    
    def __init__(self, device: str = None):
        """
        Initialize trOCR adapter.
        
        Args:
            device: Device to run on. Options: "auto", "cpu", "cuda", "mps".
                    If None, uses "cuda" if available, otherwise "cpu".
        """
        if not TROCR_AVAILABLE:
            raise ImportError(
                "trOCR dependencies not installed. Install with: "
                "pip install transformers torch"
            )
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self._processor = None
        self._model = None
        self._tokenizer = None
    
    def _init_model(self):
        """Load model, tokenizer, and processor."""
        if self._model is not None:
            return
        
        print(f"Loading trOCR model ({self.MODEL_NAME}) on {self.device}...")
        
        from transformers import ViTImageProcessor
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        
        # Load model with appropriate device mapping
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.MODEL_NAME,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load image processor (ViTFeatureExtractor is deprecated, use ViTImageProcessor)
        image_processor = ViTImageProcessor.from_pretrained(self.MODEL_NAME)
        
        # Load processor with explicit image processor to handle deprecated type
        self._processor = TrOCRProcessor(image_processor=image_processor, tokenizer=self._tokenizer)
        
        # Move model to device if not already handled by device_map
        if self.device == "cpu" and not hasattr(self._model, "device"):
            self._model.to("cpu")
        elif self.device in ["cuda", "mps"] and not self._model.device.type == self.device:
            self._model.to(self.device)
        
        print(f"trOCR model loaded on {self.device}")
    
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
        self._init_model()
        
        import time
        start_time = time.perf_counter()
        
        try:
            # Convert image to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Preprocess image
            pixel_values = self._processor(image, return_tensors="pt").pixel_values
            
            # Move to device
            pixel_values = pixel_values.to(self._model.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self._model.generate(pixel_values)
            
            # Decode
            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            latency = time.perf_counter() - start_time
            
            stats = {
                "latency_sec": latency,
                "device": self.device,
                "text_length": len(generated_text),
            }
            
            return generated_text, stats
            
        except Exception as e:
            latency = time.perf_counter() - start_time
            return "", {
                "latency_sec": latency,
                "error": str(e),
            }
