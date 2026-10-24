"""Turn raw JSONL logs into tidy tables.

trials.csv : one row per (arm, trial)          - for shares, accuracy, failure rates
long.csv   : one row per (arm, trial, room)    - for the conditional logit (chosen = 0/1)

An "arm" is model x interface x reasoning depth (e.g. ollama-qwen2.5-7b__text__low).
Files are written as UTF-8 with BOM so Excel shows ₹ and emoji correctly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hotel_lab import RUNS_DIR
from hotel_lab.runner import read_jsonl

FACTORS = ("order", "style", "review_set", "format")


def load_raw(experiment: str) -> list[dict]:
    raw_dir = RUNS_DIR / experiment / "raw"
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No raw logs in {raw_dir}. Run trials first:  python -m hotel_lab run --models ...")
    latest: dict[tuple, dict] = {}
    for p in files:
        for rec in read_jsonl(p):
            if "trial_id" not in rec:
                continue
            arm = f"{rec['model_name']}__{rec.get('interface', 'text')}__{rec.get('reasoning_depth', 'low')}"
            rec["arm"] = arm
            key = (arm, rec["trial_id"])
            # keep the latest record; a successful record is never replaced by an API-error record
            if key in latest and rec.get("error") and not latest[key].get("error"):
                continue
            latest[key] = rec
    return list(latest.values())


def export(experiment: str) -> tuple[Path, Path]:
    recs = load_raw(experiment)
    trial_rows, long_rows = [], []
    for r in recs:
        base = {k: r.get(k) for k in ("arm", "model_name", "family", "size_b", "quantisation", "serving", "interface",
                                       "reasoning_depth", "trial_id", "block", "cell", "scenario_id", "item",
                                       "repeat", "prompt_id", "retest", "focal_room_id", "best_room_id", "seed")}
        base["retest"] = bool(base.get("retest"))
        base["item_id"] = f"{r['scenario_id']}__i{r['item']}"
        api_error = bool(r.get("error"))
        trial_rows.append({
            **base, "api_error": api_error, "parse_ok": bool(r.get("parse_ok")), "attempts": r.get("attempts"),
            "first_attempt_ok": r.get("first_attempt_ok"),
            "chosen_room_id": r.get("chosen_room_id"),
            "chosen_room_uid": f"{r['scenario_id']}:{r['chosen_room_id']}" if r.get("chosen_room_id") else None,
            "chosen_position": r.get("parsed_option"),
            "chose_best": r.get("chose_best"),
            "chose_focal": (r.get("chosen_room_id") == r.get("focal_room_id")) if r.get("chosen_room_id") else None,
            "name_mismatch": r.get("name_mismatch"), "finish_reason": r.get("finish_reason"),
            "possible_truncation": r.get("possible_truncation"), "reason": r.get("reason"),
            "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"),
            "error": r.get("error"),
            **{f"cell_{k}": v for k, v in (r.get("cell_levels") or {}).items()},
        })
        if api_error or not r.get("parse_ok"):
            continue
        shells = r.get("shell_map") or {}
        first = r.get("first_review_rating") or {}
        for pos, rid in enumerate(r["room_order"], start=1):
            lv = r["room_levels"][rid]
            long_rows.append({
                **base, "choice_set_id": f"{r['arm']}::{r['trial_id']}",
                "room_id": rid, "room_uid": f"{r['scenario_id']}:{rid}",
                "shell_uid": f"{r['scenario_id']}:{shells.get(rid, rid)}", "position": pos,
                "chosen": int(rid == r["chosen_room_id"]), "is_best": int(rid == r["best_room_id"]),
                "is_focal": int(rid == r["focal_room_id"]), "first_review_rating": first.get(rid),
                **{k: lv.get(k) for k in FACTORS},
            })
    out_dir = RUNS_DIR / experiment / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_path, l_path = out_dir / "trials.csv", out_dir / "long.csv"
    pd.DataFrame(trial_rows).to_csv(t_path, index=False, encoding="utf-8-sig")
    base_cols = ["arm", "model_name", "family", "size_b", "quantisation", "serving", "interface", "reasoning_depth",
                 "trial_id", "block", "cell", "scenario_id", "item", "repeat", "prompt_id", "retest",
                 "focal_room_id", "best_room_id", "seed"]
    long_cols = base_cols + ["item_id", "choice_set_id", "room_id", "room_uid", "shell_uid", "position", "chosen",
                              "is_best", "is_focal", "first_review_rating", *FACTORS]
    pd.DataFrame(long_rows, columns=list(dict.fromkeys(long_cols))).to_csv(l_path, index=False, encoding="utf-8-sig")
    return t_path, l_path
