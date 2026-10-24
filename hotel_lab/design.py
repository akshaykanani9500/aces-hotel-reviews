"""Turn config/experiment.yaml into a reproducible trial plan (runs/<experiment>/plan.jsonl).

Vocabulary (same as the brief):
  block  - a group of trials that manipulates one factor (or a small interaction)
  cell   - one combination of factor levels inside a block
  item   - one choice set (scenario) with a given focal room
  repeat - the same item and cell again, with a fresh random position shuffle
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

import yaml

from hotel_lab import CONFIG_DIR, RUNS_DIR
from hotel_lab.stimuli import Scenario

FACTORS = ("order", "style", "review_set", "format")
ORDER_LEVELS = {"random", "newest", "oldest", "most_helpful", "extreme_first", "best_first", "worst_first"}
FORMAT_LEVELS = {"full", "summary", "both"}
ASSIGN_MODES = {"whole_set", "focal", "per_room"}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_DIR / "experiment.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check_config(cfg)
    return cfg


def check_config(cfg: dict[str, Any]) -> None:
    for k in ("experiment_name", "global_seed", "blocks", "defaults"):
        if k not in cfg:
            raise ValueError(f"experiment.yaml: missing '{k}'")
    for f in FACTORS:
        if f not in cfg["defaults"]:
            raise ValueError(f"experiment.yaml: defaults needs '{f}'")
    for b in cfg["blocks"]:
        for fname, spec in b["factors"].items():
            if fname not in FACTORS:
                raise ValueError(f"block {b['name']}: unknown factor '{fname}' (use {FACTORS})")
            if spec.get("assign") not in ASSIGN_MODES:
                raise ValueError(f"block {b['name']}/{fname}: assign must be one of {ASSIGN_MODES}")
            if spec["assign"] == "focal" and "control" not in spec:
                raise ValueError(f"block {b['name']}/{fname}: focal assignment needs a 'control' level")
            if fname == "order":
                bad = set(spec["levels"]) - ORDER_LEVELS
                if bad:
                    raise ValueError(f"block {b['name']}: unknown order levels {bad}")
            if fname == "format":
                bad = set(spec["levels"]) - FORMAT_LEVELS
                if bad:
                    raise ValueError(f"block {b['name']}: unknown format levels {bad}")


def used_levels(cfg: dict[str, Any], factor: str) -> list[str]:
    """All levels of a factor that the enabled blocks (and defaults) can produce."""
    lv = {cfg["defaults"][factor]}
    for b in enabled_blocks(cfg):
        spec = b["factors"].get(factor)
        if spec:
            lv |= set(spec["levels"])
            if "control" in spec:
                lv.add(spec["control"])
    return sorted(lv)


def enabled_blocks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in cfg["blocks"] if b.get("enabled", True)]


def stable_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed from any parts (same on every computer and Python version)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def cell_label(levels: dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in levels.items())


def build_plan(cfg: dict[str, Any], scenarios: list[Scenario]) -> list[dict[str, Any]]:
    if cfg.get("scenarios", "all") != "all":
        wanted = set(cfg["scenarios"])
        scenarios = [s for s in scenarios if s.id in wanted]
        missing = wanted - {s.id for s in scenarios}
        if missing:
            raise ValueError(f"experiment.yaml lists unknown scenarios {missing}")

    trials = []
    n_items = int(cfg.get("items_per_scenario", 4))
    n_rep = int(cfg.get("repeats", 1))
    prompts = cfg.get("prompt_ids", ["main"])

    for block in enabled_blocks(cfg):
        factors = block["factors"]
        names = list(factors)
        # per_room factors are drawn inside each trial, so they do not create cells
        cell_axes = [factors[n]["levels"] if factors[n]["assign"] != "per_room" else ["per_room"] for n in names]
        for combo in itertools.product(*cell_axes):
            cell_levels = dict(zip(names, combo))
            cell = cell_label(cell_levels)
            for sc in scenarios:
                room_ids = [r["room_id"] for r in sc.rooms]
                for item in range(n_items):
                    focal = room_ids[item % len(room_ids)]
                    for rep in range(n_rep):
                        for pid in prompts:
                            tid = f"{block['name']}__{cell.replace('|', '+').replace('=', '-')}__{sc.id}__i{item}__r{rep}__{pid}"
                            seed = stable_seed(cfg["global_seed"], tid)
                            rng = random.Random(seed)
                            order = room_ids[:]
                            rng.shuffle(order)  # product positions shuffled every trial
                            room_levels = {}
                            for rid in room_ids:
                                lv = dict(cfg["defaults"])
                                for fname, spec in factors.items():
                                    level = cell_levels[fname]
                                    if spec["assign"] == "whole_set":
                                        lv[fname] = level
                                    elif spec["assign"] == "focal":
                                        lv[fname] = level if rid == focal else spec["control"]
                                    else:  # per_room: independent random draw, cell label is ignored
                                        lv[fname] = rng.choice(spec["levels"])
                                room_levels[rid] = lv
                            trials.append({
                                "trial_id": tid,
                                "experiment": cfg["experiment_name"],
                                "block": block["name"],
                                "cell": cell,
                                "cell_levels": cell_levels,
                                "assign": {f: s["assign"] for f, s in factors.items()},
                                "scenario_id": sc.id,
                                "item": item,
                                "repeat": rep,
                                "prompt_id": pid,
                                "focal_room_id": focal,
                                "seed": seed,
                                "room_order": order,
                                "room_levels": room_levels,
                                "best_room_id": sc.best_room_id,
                            })
    return trials


def plan_path(experiment: str) -> Path:
    return RUNS_DIR / experiment / "plan.jsonl"


def save_plan(trials: list[dict[str, Any]], experiment: str) -> Path:
    p = plan_path(experiment)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return p


def load_plan(experiment: str) -> list[dict[str, Any]]:
    p = plan_path(experiment)
    if not p.exists():
        raise FileNotFoundError(f"No plan at {p}. Run:  python -m hotel_lab plan")
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
