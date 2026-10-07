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
# A reply cut off mid-reasoning has no closing tag: everything from <think> on is reasoning.
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
# Chat templates that put <think> in the prompt leave only the closing tag in the reply.
_THINK_CLOSE = re.compile(r"^.*</think>", re.DOTALL | re.IGNORECASE)
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


def strip_zero_width(text: str) -> str:
    """Drop ZWJ/ZWNJ/ZWSP and similar zero-width characters (invisible, rendered inconsistently)."""
    return text.translate(_ZERO_WIDTH)


# One visible Devanagari character (akshara): a chain of consonant + virama, ending in a consonant
# with its nukta, vowel signs and marks, and an optional trailing virama; or an independent vowel
# with its marks; anything else (digits, danda, Latin, space) is one unit.
_CONS = "\u0915-\u0939\u0958-\u095f"
_MARKS = "\u0900-\u0903\u093a\u093b\u093e-\u094c\u094e\u094f\u0951-\u0957\u0962\u0963"
_AKSHARA = re.compile(
    rf"(?:[{_CONS}]\u093c?\u094d)*[{_CONS}]\u093c?[{_MARKS}]*\u094d?"
    rf"|[\u0904-\u0914\u0960\u0961\u0972-\u097f][{_MARKS}]*"
    r"|.",
    re.DOTALL,
)


def aksharas(text: str) -> list[str]:
    """Split text into aksharas (what a reader sees as one character). Zero-width characters are
    dropped first, so a conjunct is one unit whether or not a ZWJ/ZWNJ was written."""
    return _AKSHARA.findall(strip_zero_width(text))


def clean_model_output(text: str) -> str:
    """Strip wrapping that is not part of the answer: reasoning blocks (closed, unclosed or
    with only the closing tag) and a code fence around the whole reply. Applied before
    scoring, recorded in the run config."""
    text = _THINK.sub("", text or "")
    text = _THINK_CLOSE.sub("", _THINK_OPEN.sub("", text)).replace("<|endoftext|>", "")
    match = _FENCE.match(text)
    if match:
        text = match.group(1)
    return text.strip()
