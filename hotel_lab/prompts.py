"""Build the task prompt the agent receives. Templates live in config/prompts/ as plain text."""
from __future__ import annotations

import json
from typing import Any

from hotel_lab import CONFIG_DIR

PROMPT_DIR = CONFIG_DIR / "prompts"
HIDDEN_KEYS = {"levels", "room_id", "shell_id"}  # never shown to the agent


def _read(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def _fill(template: str, **values: Any) -> str:
    # str.replace instead of str.format so the JSON braces in templates need no escaping
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def listings_json(view: dict[str, Any]) -> str:
    rooms = []
    for r in view["rooms"]:
        visible = {"option_number": r["option_number"]}
        visible.update({k: v for k, v in r.items() if k not in HIDDEN_KEYS and k != "option_number"})
        if "reviews" in visible:
            visible["reviews"] = [{k: v for k, v in rv.items() if k != "review_id"} for rv in visible["reviews"]]
        visible["currency"] = view["currency"]
        rooms.append(visible)
    # compact JSON: about 30% fewer tokens than indented JSON, same content
    return json.dumps(rooms, ensure_ascii=False, separators=(", ", ": "))


def build_prompt(view: dict[str, Any], prompt_id: str = "main", interface: str = "text",
                 reasoning_depth: str = "low") -> str:
    n = len(view["rooms"])
    if interface == "text":
        listings = _fill(_read("listings_text"), json=listings_json(view))
    elif interface == "screenshot":
        listings = _fill(_read("listings_screenshot"), n=n, n_plus_1=n + 1)
    else:
        raise ValueError(f"unknown interface '{interface}'")
    answer = _fill(_read(f"answer_{reasoning_depth}"), n=n)
    return _fill(_read(prompt_id), query=view["query"], check_in=view.get("check_in") or "the selected date",
                 nights=view.get("nights", 1), listings=listings, answer_format=answer)
