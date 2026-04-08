"""
Standalone scoring helpers — also available from ``engine.py``.

Extracting them here so external graders / validators can import without
pulling in the full engine state machine.
"""

from __future__ import annotations

import re
import unicodedata

from support_env.models import Category

RAW_MIN = -3.0
RAW_MAX = 1.2


def normalize_text(value: str) -> str:
    s = unicodedata.normalize("NFKC", value)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_category_from_classify_content(content: str) -> Category | None:
    n = normalize_text(content)
    for c in Category:
        token = c.value
        if n == token or token in n.split():
            return c
    if any(k in n for k in ("billing", "payment", "charge", "invoice", "refund")):
        return Category.BILLING
    if any(k in n for k in ("technical", "login", "crash", "slow", "performance", "app")):
        return Category.TECHNICAL
    if "general" in n:
        return Category.GENERAL
    return None


def normalized_episode_score(cumulative_raw: float) -> float:
    if RAW_MAX == RAW_MIN:
        return 0.0
    x = (cumulative_raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
    return max(0.0, min(1.0, x))
