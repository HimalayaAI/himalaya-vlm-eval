"""Text normalization for scoring.

Two levels, both explicit, because Devanagari makes "the same text" ambiguous:

- `canonical` — what CER/WER use. Unicode NFC (a precomposed nukta letter and its
  decomposed form are the same text), every whitespace run (incl. NBSP/U+202F) to one
  space, trimmed. Nothing a reader would call a different string is changed.
- `loose` — for exact-match-style checks: canonical, then drop zero-width (non-)joiners,
  punctuation in both scripts (। ॥ included) and case.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+", re.UNICODE)
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b‌‍⁠﻿"))
_DEVANAGARI_PUNCT = "।॥॰"  # । ॥ ॰
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _WS.sub(" ", text).strip()


def _is_punct(ch: str) -> bool:
    return ch in _DEVANAGARI_PUNCT or unicodedata.category(ch).startswith("P")


def loose(text: str) -> str:
    text = canonical(text).translate(_ZERO_WIDTH)
    text = "".join(" " if _is_punct(ch) else ch for ch in text)
    return _WS.sub(" ", text).strip().casefold()


def clean_model_output(text: str) -> str:
    """Strip wrapping that is not part of the answer: reasoning blocks and a code fence
    around the whole reply. Applied before scoring, recorded in the run config."""
    text = _THINK.sub("", text or "").replace("<|endoftext|>", "")
    match = _FENCE.match(text)
    if match:
        text = match.group(1)
    return text.strip()
