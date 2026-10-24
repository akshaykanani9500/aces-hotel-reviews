"""Parse the agent's reply. A reply we cannot read is recorded as a failure, never as a choice.

(The original ACES headless runner falls back to "product 1" on a parse failure, which would
silently add a position-1 bias. We do not do that.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

OPTION_KEYS = ("option_number", "selected_option", "chosen_option", "option", "selected_product_number", "choice")


@dataclass
class ParsedChoice:
    option_number: Optional[int]
    reason: str = ""
    hotel_name: str = ""
    ok: bool = False
    error: str = ""


def _json_objects(text: str):
    """Yield every JSON object that can be decoded starting at a '{' in the text."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _as_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        m = re.search(r"\d+", v)
        return int(m.group()) if m else None
    return None


def parse_choice(text: Optional[str], n_options: int) -> ParsedChoice:
    if not text or not text.strip():
        return ParsedChoice(None, error="empty reply")
    cleaned = re.sub(r"```(?:json)?", "", text)
    candidates = list(_json_objects(cleaned))
    # prefer the LAST object with an option key (reasoning-first prompts put the answer at the end)
    for obj in reversed(candidates):
        for k in OPTION_KEYS:
            if k in obj:
                num = _as_int(obj[k])
                if num is None:
                    return ParsedChoice(None, error=f"'{k}' is not a number: {obj[k]!r}")
                if not 1 <= num <= n_options:
                    return ParsedChoice(None, error=f"option {num} out of range 1-{n_options}")
                return ParsedChoice(num, str(obj.get("reason", "")), str(obj.get("hotel_name", "")), True)
    m = re.search(r'"?option_number"?\s*[:=]\s*"?(\d+)', cleaned)
    if m:
        num = int(m.group(1))
        if 1 <= num <= n_options:
            return ParsedChoice(num, "", "", True, error="recovered with regex (JSON was malformed)")
    return ParsedChoice(None, error="no option_number found")


def name_matches(claimed: str, actual: str) -> Optional[bool]:
    """Did the agent name the hotel at the option it picked? None if it gave no name."""
    if not claimed:
        return None
    a, b = claimed.strip().lower(), actual.strip().lower()
    return a in b or b in a
