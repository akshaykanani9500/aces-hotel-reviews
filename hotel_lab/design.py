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
    """Deterministic seed from any parts (same on every computer and Python version).
    Kept below 2**31 - 1000 so it fits servers that expect a signed 32-bit seed (retries add a little)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1000)


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
    n_items = int(cfg.get("items_per_scenario", 8))
    n_rep = int(cfg.get("repeats", 1))
    prompts = cfg.get("prompt_ids", ["main"])
    rotate = bool(cfg.get("rotate_shells", True))
    balance = cfg.get("position_balance", "latin")
    g = cfg["global_seed"]

    for block in enabled_blocks(cfg):
        factors = block["factors"]
        names = list(factors)
        copies = [False, True] if block.get("retest", False) else [False]
        # per_room factors are drawn inside each trial, so they do not create cells
        cell_axes = [factors[n]["levels"] if factors[n]["assign"] != "per_room" else ["per_room"] for n in names]
        for combo in itertools.product(*cell_axes):
            cell_levels = dict(zip(names, combo))
            cell = cell_label(cell_levels)
            for sc in scenarios:
                room_ids = [r["room_id"] for r in sc.rooms]
                n = len(room_ids)
                for item in range(n_items):
                    focal_idx = item % n
                    focal = room_ids[focal_idx]
                    # Item = which room is focal + which "shell" each fact profile wears.
                    # A shell is the visible hotel identity (name, location, price, size ...). Rotating
                    # shells across items stops accuracy being confounded with price or location.
                    offset = ((item // n) + item) % n if rotate else 0
                    shell_map = {rid: room_ids[(i + offset) % n] for i, rid in enumerate(room_ids)}
                    # Layout depends on (scenario, item, repeat) only, NOT the cell, so the same item
                    # and repeat has the same positions in every cell (paired comparisons, e.g. H1).
                    base = room_ids[:]
                    random.Random(stable_seed(g, sc.id, item, "layout")).shuffle(base)
                    for rep in range(n_rep):
                        layout_seed = stable_seed(g, sc.id, item, rep, "layout")
                        if balance == "latin":   # cyclic Latin square: each room in each position once per n repeats
                            k = rep % n
                            order = base[k:] + base[:k]
                        else:
                            order = room_ids[:]
                            random.Random(layout_seed).shuffle(order)
                        for pid in prompts:
                            for retest in copies:
                                tid = f"{block['name']}__{cell.replace('|', '+').replace('=', '-')}__{sc.id}__i{item}__r{rep}__{pid}"
                                if retest:
                                    tid += "__retest"
                                seed = stable_seed(g, tid)           # model sampling seed
                                rng = random.Random(seed)
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
                                    "retest": retest,
                                    "focal_room_id": focal,
                                    "seed": seed,
                                    "layout_seed": layout_seed,
                                    "room_order": order,
                                    "shell_map": shell_map,
                                    "room_levels": room_levels,
                                    "best_room_id": sc.best_room_id,
                                })
    return trials


def manifest(cfg_path: Path, scenarios: list[Scenario]) -> dict[str, Any]:
    """Fingerprints of everything that defines the study (stimuli, config, code version)."""
    import subprocess
    from hotel_lab import REPO_ROOT, __version__

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "hotel_lab"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        commit, dirty = None, None
    m = {"hotel_lab_version": __version__, "git_commit": commit, "uncommitted_changes": dirty,
         "config_sha": sha(Path(cfg_path)), "stimuli_sha": {s.id: sha(s.path) for s in scenarios}}
    m["manifest_sha"] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]
    return m


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
