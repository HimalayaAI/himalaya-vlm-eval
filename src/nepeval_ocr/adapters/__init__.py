from .base import BaseOCRAdapter
from .tesseract import TesseractAdapter
from .openai_api import OpenAIVisionAdapter
from .surya import SuryaAdapter
from .trocr import TrOCRAdapter
from .easyocr import EasyOCRAdapter
from .paddle import PaddleOCRAdapter

__all__ = ["BaseOCRAdapter", "TesseractAdapter", "OpenAIVisionAdapter", "SuryaAdapter", "TrOCRAdapter", "EasyOCRAdapter", "PaddleOCRAdapter"]
