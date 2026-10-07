"""Classical / specialised OCR engines run in-process. Transcription only.

Each engine imports its heavy dependency in `setup()`, so listing the catalog or running
another model never needs it installed. Install with the matching extra, e.g.
`pip install -e '.[tesseract]'`.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from ..types import Generation, Prompt
from .base import FatalModelError, Model, ModelError


def _auto_device(device: str | None) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _missing(extra: str, exc: ImportError) -> FatalModelError:
    return FatalModelError(f"{exc}. Install with: pip install -e '.[{extra}]'")


class _Engine(Model):
    kind = "ocr_engine"
    tasks = frozenset({"ocr"})
    max_concurrency = 1

    def generate(self, image: Any, prompt: Prompt) -> Generation:
        started = time.perf_counter()
        try:
            text = self._recognize(image)
        except (FatalModelError, ModelError):
            raise
        except Exception as exc:  # engine-specific failures on one image
            raise ModelError(f"{type(exc).__name__}: {exc}") from exc
        return Generation(text=text, latency_s=time.perf_counter() - started)

    def _recognize(self, image: Any) -> str:
        raise NotImplementedError


class TesseractModel(_Engine):
    def __init__(self, lang: str = "nep", psm: int | None = None, tessdata_dir: str | None = None):
        self.lang, self.psm, self.tessdata_dir = lang, psm, tessdata_dir
        self.max_concurrency = os.cpu_count() or 1  # one process per call; parallelises well

    def describe(self) -> dict[str, Any]:
        return {"adapter": "tesseract", "lang": self.lang, "psm": self.psm}

    def setup(self) -> None:
        try:
            import pytesseract
        except ImportError as exc:
            raise _missing("tesseract", exc) from exc
        self._pt = pytesseract
        try:
            version = pytesseract.get_tesseract_version()
        except Exception as exc:
            raise FatalModelError(f"tesseract binary not found: {exc}") from exc
        self.version = str(version)
        langs = set(pytesseract.get_languages(config=self._config()))
        missing = [lang for lang in self.lang.split("+") if lang not in langs]
        if missing:
            raise FatalModelError(
                f"tesseract language data missing: {missing}. Install e.g. tesseract-data-nep, "
                "or set tessdata_dir (this repo ships tessdata/nep.traineddata)."
            )

    def _config(self) -> str:
        parts = []
        if self.psm is not None:
            parts.append(f"--psm {self.psm}")
        if self.tessdata_dir:
            parts.append(f'--tessdata-dir "{self.tessdata_dir}"')
        return " ".join(parts)

    def _recognize(self, image: Any) -> str:
        return self._pt.image_to_string(image, lang=self.lang, config=self._config()).strip()


class EasyOCRModel(_Engine):
    def __init__(self, langs: list[str] | None = None, device: str | None = None):
        self.langs = langs or ["ne", "en"]
        self.device = device

    def describe(self) -> dict[str, Any]:
        return {"adapter": "easyocr", "langs": self.langs}

    def setup(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise _missing("easyocr", exc) from exc
        self.device = _auto_device(self.device)
        self._reader = easyocr.Reader(lang_list=self.langs, gpu=self.device == "cuda")

    def _recognize(self, image: Any) -> str:
        import numpy as np

        # paragraph=True orders detections into reading order before joining
        results = self._reader.readtext(np.array(image.convert("RGB")), detail=0, paragraph=True)
        return "\n".join(r.strip() for r in results if r and r.strip())


class PaddleOCRModel(_Engine):
    def __init__(self, lang: str = "ne", ocr_version: str = "PP-OCRv5", device: str | None = None):
        self.lang, self.ocr_version, self.device = lang, ocr_version, device

    def describe(self) -> dict[str, Any]:
        return {"adapter": "paddle", "lang": self.lang, "ocr_version": self.ocr_version}

    def setup(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise _missing("paddle", exc) from exc
        kwargs: dict[str, Any] = {"lang": self.lang, "ocr_version": self.ocr_version}
        if self.device:
            kwargs["device"] = "gpu" if self.device == "cuda" else self.device
        try:  # PaddleOCR 3.x
            self._ocr = PaddleOCR(use_textline_orientation=True, **kwargs)
            self._v3 = True
        except TypeError:  # 2.x signature
            kwargs.pop("device", None)
            self._ocr = PaddleOCR(use_angle_cls=True, show_log=False, **kwargs)
            self._v3 = False

    def _recognize(self, image: Any) -> str:
        import numpy as np

        arr = np.array(image.convert("RGB"))
        if self._v3:
            lines: list[str] = []
            for page in self._ocr.predict(arr) or []:
                data = page.json.get("res", page.json) if hasattr(page, "json") else page
                lines.extend(data.get("rec_texts", []))
            return "\n".join(t for t in lines if t)
        result = self._ocr.ocr(arr, cls=True) or []
        lines = [item[1][0] for page in result if page for item in page]
        return "\n".join(lines)


class SuryaModel(_Engine):
    """Surya with its own inference server (vllm on GPU, llama.cpp on CPU)."""

    def __init__(self, backend: str | None = None):
        self.backend = backend or os.environ.get("SURYA_INFERENCE_BACKEND", "llamacpp")

    def describe(self) -> dict[str, Any]:
        return {"adapter": "surya", "backend": self.backend}

    def setup(self) -> None:
        os.environ["SURYA_INFERENCE_BACKEND"] = self.backend
        try:
            import surya.settings

            surya.settings.settings.SURYA_INFERENCE_BACKEND = self.backend
            if self.backend == "llamacpp":  # force local inference, not a remote URL
                os.environ.pop("SURYA_INFERENCE_URL", None)
                surya.settings.settings.SURYA_INFERENCE_URL = None
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise _missing("surya", exc) from exc
        try:
            self._predictor = RecognitionPredictor(SuryaInferenceManager())
        except Exception as exc:
            raise FatalModelError(
                f"Surya backend failed to start ({exc}). GPU: Docker with NVIDIA runtime; "
                "CPU: llama.cpp `llama-server` on PATH."
            ) from exc

    def _recognize(self, image: Any) -> str:
        pages = self._predictor([image.convert("RGB")])
        if not pages:
            return ""

        def html_to_text(html: str) -> str:
            return re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", "\n", html or ""))

        return "\n".join(html_to_text(b.html).strip() for b in pages[0].blocks if b.html)


class TrOCRModel(_Engine):
    def __init__(self, checkpoint: str = "syubraj/TrOCR_Nepali", device: str | None = None,
                 max_new_tokens: int = 256):
        self.checkpoint, self.device, self.max_new_tokens = checkpoint, device, max_new_tokens

    def describe(self) -> dict[str, Any]:
        return {"adapter": "trocr", "checkpoint": self.checkpoint,
                "max_new_tokens": self.max_new_tokens}

    def setup(self) -> None:
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                TrOCRProcessor,
                VisionEncoderDecoderModel,
                ViTImageProcessor,
            )
        except ImportError as exc:
            raise _missing("trocr", exc) from exc
        self._torch = torch
        self.device = _auto_device(self.device)
        tok = AutoTokenizer.from_pretrained(self.checkpoint)
        self._processor = TrOCRProcessor(
            image_processor=ViTImageProcessor.from_pretrained(self.checkpoint), tokenizer=tok
        )
        self._model = VisionEncoderDecoderModel.from_pretrained(self.checkpoint).to(self.device)
        self._model.eval()

    def _recognize(self, image: Any) -> str:
        pixels = self._processor(image.convert("RGB"), return_tensors="pt").pixel_values
        with self._torch.no_grad():
            ids = self._model.generate(pixels.to(self.device), max_new_tokens=self.max_new_tokens)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0]
