"""One study = the Google Sheet (or YAML fallback) + one experiment per hypothesis block.

  prepare(block)  -> builds the trial plan for that block only (one factor at a time; every other factor
                     sits at its control/default level), saved under runs/<study>__<block>/
  run(block, ...) -> runs the selected models, logs every trial locally, pushes rows to the Sheet
  analyze()       -> exports every block, computes all verdicts, writes them locally and to the Sheet
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import yaml

from hotel_lab import RUNS_DIR
from hotel_lab import sheetdb
from hotel_lab.design import build_plan, check_config, manifest, save_plan
from hotel_lab.stimuli import load_scenarios


def load_study() -> dict[str, Any]:
    """Scenarios, hypotheses and settings from the synced snapshot, or built-in defaults."""
    snap = sheetdb.current_snapshot()
    if snap:
        tabs = snap["tabs"]
        hyps = sheetdb.tabs_to_hypotheses(tabs) or [dict(h, enabled=True) for h in sheetdb.HYPOTHESES]
        settings = sheetdb.tabs_to_settings(tabs)
        source = {"type": "google_sheet" if snap["source"] == "google_sheet" else snap["source"],
                  "sha": snap["sha"], "fetched_at": snap["fetched_at"], "validation": snap.get("validation", {})}
    else:
        hyps = [dict(h, enabled=str(h.get("enabled", "TRUE")).upper() == "TRUE") for h in sheetdb.HYPOTHESES]
        settings = {k: v for k, v, _ in sheetdb.DEFAULT_SETTINGS}
        source = {"type": "yaml (hotel_lab/stimuli)", "sha": None, "fetched_at": None, "validation": {}}
    return {"scenarios": load_scenarios(), "hypotheses": hyps, "settings": settings, "source": source}


def _levels(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def blocks(hypotheses: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for h in hypotheses:
        if h.get("enabled", True) and h.get("block"):
            out.setdefault(h["block"], []).append(h)
    return out


def block_config(block: str, study: dict[str, Any]) -> dict[str, Any]:
    """experiment.yaml-equivalent for one block, built from the Hypotheses + Settings tabs."""
    hs = blocks(study["hypotheses"]).get(block)
    if not hs:
        raise KeyError(f"No enabled hypothesis uses block '{block}'")
    h = hs[0]
    s = study["settings"]
    factors = {}
    for f, lv, asg, ctl in (("factor", "levels", "assign", "control"), ("factor2", "levels2", "assign2", "control2")):
        if h.get(f):
            spec = {"levels": _levels(h[lv]), "assign": h.get(asg) or "whole_set"}
            if h.get(ctl):
                spec["control"] = h[ctl]
            factors[h[f]] = spec
    cfg = {
        "experiment_name": f"{s.get('experiment_name', 'study_v1')}__{block}",
        "global_seed": int(s.get("global_seed", 20261110)),
        "scenarios": "all",
        "items_per_scenario": int(s.get("items_per_scenario", 4)),
        "repeats": int(s.get("repeats", 8)),
        "rotate_shells": str(s.get("rotate_shells", "TRUE")).upper() == "TRUE",
        "identical_rooms": str(s.get("identical_rooms", "TRUE")).upper() == "TRUE",
        "position_balance": s.get("position_balance", "latin"),
        "prompt_ids": _levels(s.get("prompt_ids", "main")) or ["main"],
        "interface": s.get("interface", "text"),
        "reasoning_depth": s.get("reasoning_depth", "low"),
        "parse_retries": int(s.get("parse_retries", 1)),
        "defaults": {"order": s.get("default_order", "random"), "style": s.get("default_style", "base"),
                     "review_set": s.get("default_review_set", "base"), "format": s.get("default_format", "full"),
                     "facts": "real", "placebo": "A"},
        "blocks": [{"name": block, "enabled": True,
                    "retest": any(str(x.get("retest", "")).upper() == "TRUE" for x in hs), "factors": factors}],
        "show_category_scores": False,
        "hypotheses": [x["hypothesis_id"] for x in hs],
    }
    check_config(cfg)
    return cfg


def prepare(block: str, study: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    study = study or load_study()
    cfg = block_config(block, study)
    trials = build_plan(cfg, study["scenarios"])
    # pre-flight: every trial must render
    from hotel_lab.conditions import build_view
    sc = {s.id: s for s in study["scenarios"]}
    problems = {}
    for t in trials:
        try:
            build_view(sc[t["scenario_id"]], t)
        except (KeyError, ValueError) as e:
            problems.setdefault(str(e), t["trial_id"])
    if problems:
        raise ValueError("Cannot build this experiment: " + "; ".join(list(problems)[:3]))
    exp = cfg["experiment_name"]
    p = save_plan(trials, exp)
    with open(p.parent / "experiment.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    cfg_path = p.parent / "experiment.yaml"
    man = manifest(cfg_path, study["scenarios"])
    man["snapshot_sha"] = study["source"].get("sha")
    (p.parent / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    cells = sorted({t["cell"] for t in trials})
    main = [t for t in trials if not t.get("retest")]
    return {"experiment": exp, "block": block, "trials": len(trials), "cells": cells,
            "per_cell": len(main) // max(len(cells), 1), "retest_copies": len(trials) - len(main),
            "scenarios": len(study["scenarios"]), "hypotheses": cfg["hypotheses"]}


def result_row(rec: dict[str, Any], run_id: str, snapshot_sha: Optional[str]) -> dict[str, Any]:
    """One compact row for the Sheet's Results tab (full prompts stay in the local log)."""
    return {
        "timestamp": rec.get("timestamp"), "run_id": run_id, "experiment": rec.get("experiment"),
        "hypothesis_block": rec.get("block"), "trial_id": rec.get("trial_id"), "cell": rec.get("cell"),
        "scenario_id": rec.get("scenario_id"), "item": rec.get("item"), "repeat": rec.get("repeat"),
        "prompt_id": rec.get("prompt_id"), "retest": rec.get("retest"), "model_name": rec.get("model_name"),
        "model": rec.get("model"), "interface": rec.get("interface"), "reasoning_depth": rec.get("reasoning_depth"),
        "temperature": rec.get("temperature"), "focal_room_id": rec.get("focal_room_id"),
        "best_room_id": rec.get("best_room_id"), "room_order": " ".join(rec.get("room_order") or []),
        "chosen_option": rec.get("parsed_option"), "chosen_room_id": rec.get("chosen_room_id"),
        "chose_best": rec.get("chose_best"),
        "chose_focal": (rec.get("chosen_room_id") == rec.get("focal_room_id")) if rec.get("chosen_room_id") else None,
        "parse_ok": rec.get("parse_ok"), "attempts": rec.get("attempts"), "error": rec.get("error") or "",
        "reason": (rec.get("reason") or "")[:500], "input_tokens": rec.get("input_tokens"),
        "output_tokens": rec.get("output_tokens"), "latency_s": rec.get("latency_s"), "seed": rec.get("seed"),
        "snapshot_sha": snapshot_sha or "",
    }


def push_pending(experiment: str) -> dict[str, Any]:
    """Send rows that could not be pushed earlier (no internet, sheet not linked yet)."""
    out = RUNS_DIR / experiment / "sheet_outbox.jsonl"
    db = sheetdb.get_db()
    if not out.exists() or db is None:
        return {"pushed": 0, "pending": _count(out)}
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = 0
    for i in range(0, len(rows), 200):
        n += db.append("Results", rows[i:i + 200])
    out.unlink()
    return {"pushed": n, "pending": 0}


def _count(p) -> int:
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0


class SheetPusher:
    """Buffers Results rows and sends them in batches; anything that fails waits in the outbox."""

    def __init__(self, experiment: str, run_id: str, snapshot_sha: Optional[str], enabled: bool, batch: int = 20):
        self.experiment, self.run_id, self.sha, self.batch = experiment, run_id, snapshot_sha, batch
        self.db = sheetdb.get_db() if enabled else None
        self.buf: list[dict[str, Any]] = []
        self.pushed = 0
        self.last_error = None
        self.outbox = RUNS_DIR / experiment / "sheet_outbox.jsonl"

    def add(self, rec: dict[str, Any]) -> None:
        self.buf.append(result_row(rec, self.run_id, self.sha))
        if len(self.buf) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        rows, self.buf = self.buf, []
        if self.db is not None:
            try:
                self.pushed += self.db.append("Results", rows)
                return
            except Exception as e:
                self.last_error = str(e)
        self.outbox.parent.mkdir(parents=True, exist_ok=True)
        with open(self.outbox, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")


def analyze(study: Optional[dict[str, Any]] = None, push: bool = True) -> dict[str, Any]:
    from hotel_lab.export import export
    from hotel_lab.verdicts import evaluate
    study = study or load_study()
    name = study["settings"].get("experiment_name", "study_v1")
    for block in blocks(study["hypotheses"]):
        exp_dir = RUNS_DIR / f"{name}__{block}"
        if (exp_dir / "raw").exists() and any((exp_dir / "raw").glob("*.jsonl")):
            export(f"{name}__{block}")
    rows = evaluate(name, study["hypotheses"], study["settings"])
    out_dir = RUNS_DIR / f"{name}__verdicts"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "verdicts.csv", index=False, encoding="utf-8-sig")
    pushed = None
    if push and sheetdb.get_db() is not None:
        try:
            clean = [{k: ("" if (isinstance(v, float) and v != v) else v) for k, v in r.items()} for r in rows]
            pushed = sheetdb.get_db().replace("Verdicts", clean, sheetdb.VERDICT_COLUMNS + ["notes", "max_room_share"])
        except Exception as e:
            pushed = f"failed: {e}"
    return {"rows": rows, "csv": str(out_dir / "verdicts.csv"), "pushed": pushed}


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
