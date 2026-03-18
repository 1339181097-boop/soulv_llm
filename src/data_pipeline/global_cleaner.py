from __future__ import annotations

import re
from typing import Any

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s]{6,}\d)(?!\d)")
ID_PATTERN = re.compile(r"(?<!\d)(?:\d{15}|\d{17}[\dXx])(?!\d)")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_PATTERN.sub("", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    text = MULTI_BLANK_LINE_PATTERN.sub("\n\n", text)
    return text.strip()


def mask_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[EMAIL]", text)
    text = PHONE_PATTERN.sub("[PHONE]", text)
    text = ID_PATTERN.sub("[ID_CARD]", text)
    return text


def truncate_text(text: str, max_length: int = 12000) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "\n\n[TRUNCATED]"


def clean_text(value: Any, *, max_length: int = 12000, mask_sensitive: bool = True) -> str:
    text = normalize_text(value)
    if mask_sensitive:
        text = mask_pii(text)
    return truncate_text(text, max_length=max_length)
