"""Turn raw JSONL logs into tidy tables.

trials.csv : one row per (model, trial)          - for shares, accuracy, failure rates
long.csv   : one row per (model, trial, room)    - for the conditional logit (chosen = 0/1)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hotel_lab import RUNS_DIR

FACTORS = ("order", "style", "review_set", "format")


def load_raw(experiment: str) -> list[dict]:
    raw_dir = RUNS_DIR / experiment / "raw"
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No raw logs in {raw_dir}. Run trials first.")
    latest: dict[tuple[str, str], dict] = {}
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (rec["model_name"], rec["trial_id"])
                # keep the latest record; a successful record is never replaced by an API-error record
                if key in latest and rec.get("error") and not latest[key].get("error"):
                    continue
                latest[key] = rec
    return list(latest.values())


def export(experiment: str) -> tuple[Path, Path]:
    recs = load_raw(experiment)
    trial_rows, long_rows = [], []
    for r in recs:
        base = {k: r.get(k) for k in ("model_name", "family", "size_b", "quantisation", "serving", "trial_id",
                                       "block", "cell", "scenario_id", "item", "repeat", "prompt_id",
                                       "focal_room_id", "best_room_id", "interface", "reasoning_depth", "seed")}
        base["item_id"] = f"{r['scenario_id']}__i{r['item']}"
        api_error = bool(r.get("error"))
        trial_rows.append({
            **base, "api_error": api_error, "parse_ok": bool(r.get("parse_ok")), "attempts": r.get("attempts"),
            "chosen_room_id": r.get("chosen_room_id"), "chosen_position": r.get("parsed_option"),
            "chose_best": r.get("chose_best"),
            "chose_focal": (r.get("chosen_room_id") == r.get("focal_room_id")) if r.get("chosen_room_id") else None,
            "name_mismatch": r.get("name_mismatch"), "reason": r.get("reason"),
            "input_tokens": r.get("input_tokens"), "output_tokens": r.get("output_tokens"),
            **{f"cell_{k}": v for k, v in (r.get("cell_levels") or {}).items()},
        })
        if api_error or not r.get("parse_ok"):
            continue
        for pos, rid in enumerate(r["room_order"], start=1):
            lv = r["room_levels"][rid]
            long_rows.append({
                **base, "choice_set_id": f"{r['model_name']}::{r['trial_id']}",
                "room_id": rid, "room_uid": f"{r['scenario_id']}:{rid}", "position": pos,
                "chosen": int(rid == r["chosen_room_id"]), "is_best": int(rid == r["best_room_id"]),
                "is_focal": int(rid == r["focal_room_id"]),
                **{k: lv.get(k) for k in FACTORS},
            })
    out_dir = RUNS_DIR / experiment / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_path, l_path = out_dir / "trials.csv", out_dir / "long.csv"
    pd.DataFrame(trial_rows).to_csv(t_path, index=False)
    pd.DataFrame(long_rows).to_csv(l_path, index=False)
    return t_path, l_path
