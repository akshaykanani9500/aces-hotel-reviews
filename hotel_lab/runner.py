"""Run planned trials against one or more models and log every trial (append-only JSONL).

Output: runs/<experiment>/raw/<model_name>.jsonl  - one line per attempt-complete trial.
Re-running skips trials that already have a record without an API error, so a crashed run
can simply be started again.
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from hotel_lab import RUNS_DIR, __version__
from hotel_lab.conditions import build_view
from hotel_lab.design import load_plan
from hotel_lab.models import ModelClient, ModelConfig
from hotel_lab.parse import name_matches, parse_choice
from hotel_lab.prompts import build_prompt
from hotel_lab.stimuli import load_scenarios


def raw_path(experiment: str, model_name: str) -> Path:
    return RUNS_DIR / experiment / "raw" / f"{model_name}.jsonl"


def done_trial_ids(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if not rec.get("error"):
                        done.add(rec["trial_id"])
    return done


def select_trials(trials: list[dict[str, Any]], blocks: Optional[list[str]] = None,
                  scenarios: Optional[list[str]] = None, limit: Optional[int] = None) -> list[dict[str, Any]]:
    out = [t for t in trials if (not blocks or t["block"] in blocks) and (not scenarios or t["scenario_id"] in scenarios)]
    return out[:limit] if limit else out


def run_trial(client: ModelClient, trial: dict[str, Any], view: dict[str, Any], interface: str,
              reasoning_depth: str, parse_retries: int, screenshot_paths: Optional[list[Path]] = None) -> dict[str, Any]:
    prompt = build_prompt(view, trial["prompt_id"], interface, reasoning_depth)
    images = [p.read_bytes() for p in screenshot_paths] if screenshot_paths else None
    n = len(view["rooms"])
    attempts, raw_outputs = [], []
    rec: dict[str, Any] = {"error": None}
    t_in = t_out = 0
    latency = 0.0
    parsed = None
    for attempt in range(parse_retries + 1):
        # each retry gets a different (but still deterministic) seed
        comp = client.complete(prompt, images=images, seed=trial["seed"] + attempt, n_options=n)
        raw_outputs.append(comp.text)
        t_in += comp.input_tokens or 0
        t_out += comp.output_tokens or 0
        latency += comp.latency_s
        parsed = parse_choice(comp.text, n)
        attempts.append({"ok": parsed.ok, "error": parsed.error})
        if parsed.ok:
            break

    chosen_room = None
    mismatch = None
    if parsed and parsed.ok:
        room = view["rooms"][parsed.option_number - 1]
        chosen_room = room["room_id"]
        match = name_matches(parsed.hotel_name, room["hotel_name"])
        mismatch = None if match is None else (not match)

    rec.update({
        "prompt": prompt,
        "screenshots": [str(p) for p in screenshot_paths] if screenshot_paths else None,
        "raw_output": raw_outputs[-1] if raw_outputs else None,
        "raw_outputs_all": raw_outputs,
        "attempts": len(attempts),
        "attempt_log": attempts,
        "parse_ok": bool(parsed and parsed.ok),
        "parse_note": parsed.error if parsed else "",
        "parsed_option": parsed.option_number if parsed else None,
        "chosen_room_id": chosen_room,
        "chose_best": (chosen_room == trial["best_room_id"]) if chosen_room else None,
        "reason": parsed.reason if parsed else "",
        "claimed_hotel_name": parsed.hotel_name if parsed else "",
        "name_mismatch": mismatch,
        "input_tokens": t_in or None,
        "output_tokens": t_out or None,
        "latency_s": round(latency, 3),
    })
    return rec


def run(experiment: str, model_names: Iterable[str], interface: str = "text", reasoning_depth: str = "low",
        parse_retries: int = 1, blocks: Optional[list[str]] = None, scenarios: Optional[list[str]] = None,
        limit: Optional[int] = None, show_category_scores: bool = False, quiet: bool = False) -> dict[str, Path]:
    trials = select_trials(load_plan(experiment), blocks, scenarios, limit)
    sc_map = {s.id: s for s in load_scenarios()}
    outputs = {}

    for mname in model_names:
        cfg = ModelConfig.load(mname)
        if interface == "screenshot" and cfg.provider != "mock" and not cfg.supports_vision:
            raise RuntimeError(f"{cfg.name} has supports_vision: false; screenshot mode needs a vision model")
        client = ModelClient(cfg)
        version = client.describe_version()
        out = raw_path(experiment, cfg.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = done_trial_ids(out)
        todo = [t for t in trials if t["trial_id"] not in done]
        if not quiet:
            print(f"[{cfg.name}] {len(todo)} trials to run ({len(done)} already done) -> {out}")

        site = shooter = None
        if interface == "screenshot":
            from hotel_lab.screenshots import LocalSite, Screenshotter
            site = LocalSite(experiment).__enter__()
            shooter = Screenshotter(site).__enter__()
        n_fail = 0
        try:
            with open(out, "a", encoding="utf-8") as f:
                for i, trial in enumerate(todo, 1):
                    view = build_view(sc_map[trial["scenario_id"]], trial, show_category_scores)
                    shots = None
                    if shooter:
                        shots = shooter.capture_trial(trial, RUNS_DIR / experiment / "screenshots" / trial["trial_id"])
                    base = {
                        "trial_id": trial["trial_id"], "experiment": experiment, "block": trial["block"],
                        "cell": trial["cell"], "cell_levels": trial["cell_levels"], "scenario_id": trial["scenario_id"],
                        "item": trial["item"], "repeat": trial["repeat"], "prompt_id": trial["prompt_id"],
                        "focal_room_id": trial["focal_room_id"], "seed": trial["seed"],
                        "room_order": trial["room_order"], "room_levels": trial["room_levels"],
                        "best_room_id": trial["best_room_id"],
                        "model_name": cfg.name, "model": cfg.model, "family": cfg.family, "size_b": cfg.size_b,
                        "quantisation": cfg.quantisation, "serving": cfg.serving, "model_version": version,
                        "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
                        "interface": interface, "reasoning_depth": reasoning_depth,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "hotel_lab_version": __version__, "python": platform.python_version(),
                    }
                    try:
                        rec = run_trial(client, trial, view, interface, reasoning_depth, parse_retries, shots)
                    except Exception as e:  # API/network error: logged, retried on the next run
                        rec = {"error": f"{type(e).__name__}: {e}", "parse_ok": False}
                        time.sleep(1)
                    if not rec.get("parse_ok"):
                        n_fail += 1
                    f.write(json.dumps({**base, **rec}, ensure_ascii=False) + "\n")
                    f.flush()
                    if not quiet and (i % 25 == 0 or i == len(todo)):
                        print(f"  {i}/{len(todo)} done, failures so far: {n_fail}")
        finally:
            if shooter:
                shooter.__exit__(None, None, None)
            if site:
                site.__exit__(None, None, None)
        outputs[cfg.name] = out
    return outputs


def hello(model_name: str) -> dict[str, Any]:
    """Week-1 check: can this model return a valid JSON answer?"""
    cfg = ModelConfig.load(model_name)
    client = ModelClient(cfg)
    prompt = ('Pick a number between 1 and 4. Respond ONLY with a JSON object in exactly this format: '
              '{"option_number": <1-4>, "hotel_name": "", "reason": "<one sentence>"}')
    comp = client.complete(prompt, seed=1, n_options=4)
    parsed = parse_choice(comp.text, 4)
    return {"model": cfg.name, "raw_output": comp.text, "parse_ok": parsed.ok, "option": parsed.option_number,
            "note": parsed.error, "latency_s": round(comp.latency_s, 2), "version": client.describe_version()}
