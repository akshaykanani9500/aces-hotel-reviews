"""Run planned trials against one or more models and log every trial (append-only JSONL).

Output: runs/<experiment>/raw/<model>__<interface>__<reasoning>.jsonl - one line per trial.
Text and screenshot runs (and low/high reasoning) are separate "arms" with separate files, so
running a second interface never skips or overwrites the first.
Re-running skips trials that already have a record without an API error, so a crashed run
can simply be started again.
"""
from __future__ import annotations

import dataclasses
import json
import platform
import time
from collections import defaultdict
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

FAIL_FAST_AFTER = 5          # stop if the first N trials all hit API/connection errors
HIGH_REASONING_MIN_TOKENS = 1200


def arm_name(model_name: str, interface: str, reasoning_depth: str) -> str:
    return f"{model_name}__{interface}__{reasoning_depth}"


def raw_path(experiment: str, model_name: str, interface: str = "text", reasoning_depth: str = "low") -> Path:
    return RUNS_DIR / experiment / "raw" / f"{arm_name(model_name, interface, reasoning_depth)}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a log, skipping a half-written last line (e.g. laptop slept mid-write)."""
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  note: skipped a damaged line {n} in {path.name} (probably an interrupted write)")
    return out


def done_trial_ids(path: Path) -> set[str]:
    return {r["trial_id"] for r in read_jsonl(path) if not r.get("error") and "trial_id" in r}


def select_trials(trials: list[dict[str, Any]], blocks: Optional[list[str]] = None,
                  scenarios: Optional[list[str]] = None, limit: Optional[int] = None,
                  per_cell: Optional[int] = None, include_retest: bool = True) -> list[dict[str, Any]]:
    """Filter the plan. `per_cell` = first N trials of every cell (a proper pilot).
    `limit` = N trials spread round-robin across all cells (not just the first cell)."""
    out = [t for t in trials if (not blocks or t["block"] in blocks)
           and (not scenarios or t["scenario_id"] in scenarios)
           and (include_retest or not t.get("retest"))]
    if per_cell or limit:
        cells: dict[tuple, list] = defaultdict(list)
        for t in out:
            cells[(t["block"], t["cell"])].append(t)
        if per_cell:
            out = [t for c in cells.values() for t in c[:per_cell]]
        else:
            mixed, i = [], 0
            while len(mixed) < limit and any(i < len(c) for c in cells.values()):
                mixed += [c[i] for c in cells.values() if i < len(c)]
                i += 1
            out = mixed[:limit]
    return out


def run_trial(client: ModelClient, trial: dict[str, Any], prompt: str, n_options: int, view: dict[str, Any],
              parse_retries: int, screenshot_paths: Optional[list[Path]] = None) -> dict[str, Any]:
    images = [p.read_bytes() for p in screenshot_paths] if screenshot_paths else None
    attempts, raw_outputs = [], []
    t_in = t_out = 0
    latency = 0.0
    parsed = None
    error = None
    finish = None
    meta: dict[str, Any] = {}
    for attempt in range(parse_retries + 1):
        try:
            # each retry gets a different (but still deterministic) seed
            comp = client.complete(prompt, images=images, seed=trial["seed"] + attempt, n_options=n_options)
        except Exception as e:  # API/network error: keep earlier attempts, mark the trial for re-running
            error = f"{type(e).__name__}: {e}"
            attempts.append({"ok": False, "error": error, "finish_reason": None})
            break
        raw_outputs.append(comp.text)
        t_in += comp.input_tokens or 0
        t_out += comp.output_tokens or 0
        latency += comp.latency_s
        finish, meta = comp.finish_reason, comp.response_meta
        parsed = parse_choice(comp.text, n_options)
        attempts.append({"ok": parsed.ok, "error": parsed.error, "finish_reason": comp.finish_reason,
                         "input_tokens": comp.input_tokens})
        if parsed.ok:
            break

    ok = bool(parsed and parsed.ok)
    chosen_room = mismatch = None
    if ok:
        room = view["rooms"][parsed.option_number - 1]
        chosen_room = room["room_id"]
        match = name_matches(parsed.hotel_name, room["hotel_name"])
        mismatch = None if match is None else (not match)
    first_in = attempts[0].get("input_tokens") if attempts else None
    cfg = client.cfg
    return {
        "error": error if not ok else None,
        "raw_output": raw_outputs[-1] if raw_outputs else None,
        "raw_outputs_all": raw_outputs,
        "attempts": len(attempts),
        "attempt_log": attempts,
        "first_attempt_ok": bool(attempts and attempts[0]["ok"]),
        "parse_ok": ok,
        "parse_note": parsed.error if parsed else "",
        "parsed_option": parsed.option_number if ok else None,
        "chosen_room_id": chosen_room,
        "chose_best": (chosen_room == trial["best_room_id"]) if chosen_room else None,
        "reason": parsed.reason if parsed else "",
        "claimed_hotel_name": parsed.hotel_name if parsed else "",
        "name_mismatch": mismatch,
        "finish_reason": finish,
        "response_meta": meta,
        "input_tokens": t_in or None,
        "output_tokens": t_out or None,
        # if the prompt filled the context window, the server may have cut it (Ollama cuts silently)
        "possible_truncation": bool(cfg.provider == "ollama" and first_in and first_in >= cfg.num_ctx - 8),
        "latency_s": round(latency, 3),
    }


def _manifest_sha(experiment: str) -> Optional[str]:
    p = RUNS_DIR / experiment / "manifest.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("manifest_sha")
    return None


def run(experiment: str, model_names: Iterable[str], interface: str = "text", reasoning_depth: str = "low",
        parse_retries: int = 1, blocks: Optional[list[str]] = None, scenarios: Optional[list[str]] = None,
        limit: Optional[int] = None, per_cell: Optional[int] = None, show_category_scores: bool = False,
        quiet: bool = False) -> dict[str, dict[str, Any]]:
    trials = select_trials(load_plan(experiment), blocks, scenarios, limit, per_cell)
    sc_map = {s.id: s for s in load_scenarios()}
    manifest_sha = _manifest_sha(experiment)

    # Pre-flight: render every selected trial once, so stimulus problems stop the run BEFORE any model call.
    views, prompts = {}, {}
    for t in trials:
        if t["scenario_id"] not in sc_map:
            raise SystemExit(f"Trial {t['trial_id']} uses scenario '{t['scenario_id']}', which is not in "
                             f"hotel_lab/stimuli/. Re-run `python -m hotel_lab plan`.")
        try:
            views[t["trial_id"]] = build_view(sc_map[t["scenario_id"]], t, show_category_scores)
        except (KeyError, ValueError) as e:
            raise SystemExit(f"Cannot build trial {t['trial_id']}: {e}\nFix the stimulus file or experiment.yaml, "
                             f"then re-run `python -m hotel_lab plan`.")
        prompts[t["trial_id"]] = build_prompt(views[t["trial_id"]], t["prompt_id"], interface, reasoning_depth)

    summary = {}
    for mname in model_names:
        cfg = ModelConfig.load(mname)
        if interface == "screenshot" and cfg.provider != "mock" and not cfg.supports_vision:
            raise SystemExit(f"{cfg.name} has supports_vision: false; screenshot mode needs a vision model")
        if reasoning_depth == "high":
            if cfg.max_tokens < HIGH_REASONING_MIN_TOKENS:
                cfg = dataclasses.replace(cfg, max_tokens=HIGH_REASONING_MIN_TOKENS)
                print(f"  note: reasoning_depth high -> max_tokens raised to {HIGH_REASONING_MIN_TOKENS} for {cfg.name}")
            if cfg.json_mode:
                cfg = dataclasses.replace(cfg, json_mode=False)
                print(f"  note: json_mode turned off for {cfg.name} (it would block the reasoning step)")
        if cfg.provider == "ollama" and prompts:
            longest = max(len(p) for p in prompts.values())
            est = longest / 3.2 + cfg.max_tokens
            if est > cfg.num_ctx:
                print(f"  WARNING: longest prompt is ~{int(longest / 3.2)} tokens; num_ctx={cfg.num_ctx} may be too "
                      f"small and Ollama would cut the prompt. Raise num_ctx in {cfg.name}.yaml.")
        client = ModelClient(cfg)
        version = client.describe_version()
        out = raw_path(experiment, cfg.name, interface, reasoning_depth)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = done_trial_ids(out)
        todo = [t for t in trials if t["trial_id"] not in done]
        if not quiet:
            print(f"[{arm_name(cfg.name, interface, reasoning_depth)}] {len(todo)} trials to run "
                  f"({len(done)} already done) -> {out}")

        site = shooter = None
        if interface == "screenshot" and todo:
            from hotel_lab.screenshots import LocalSite, Screenshotter
            site = LocalSite(experiment, show_category_scores=show_category_scores).__enter__()
            shooter = Screenshotter(site).__enter__()
        n_fail = n_api = n_ok = 0
        last_error = None
        try:
            with open(out, "a", encoding="utf-8") as f:
                for i, trial in enumerate(todo, 1):
                    view, prompt = views[trial["trial_id"]], prompts[trial["trial_id"]]
                    shots = None
                    if shooter:
                        shots = shooter.capture_trial(trial, RUNS_DIR / experiment / "screenshots" / trial["trial_id"])
                    base = {
                        "trial_id": trial["trial_id"], "experiment": experiment, "block": trial["block"],
                        "cell": trial["cell"], "cell_levels": trial["cell_levels"], "scenario_id": trial["scenario_id"],
                        "item": trial["item"], "repeat": trial["repeat"], "prompt_id": trial["prompt_id"],
                        "retest": trial.get("retest", False), "focal_room_id": trial["focal_room_id"],
                        "seed": trial["seed"], "layout_seed": trial.get("layout_seed"),
                        "room_order": trial["room_order"], "shell_map": trial.get("shell_map"),
                        "room_levels": trial["room_levels"], "best_room_id": trial["best_room_id"],
                        "first_review_rating": {r["room_id"]: (r["reviews"][0]["rating"] if r.get("reviews") else None)
                                                for r in view["rooms"]},
                        "model_name": cfg.name, "model": cfg.model, "provider": cfg.provider, "family": cfg.family,
                        "size_b": cfg.size_b, "quantisation": cfg.quantisation, "serving": cfg.serving,
                        "model_version": version, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
                        "num_ctx": cfg.num_ctx if cfg.provider == "ollama" else None,
                        "interface": interface, "reasoning_depth": reasoning_depth,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "hotel_lab_version": __version__, "manifest_sha": manifest_sha,
                        "python": platform.python_version(),
                        "prompt": prompt, "screenshots": [str(p) for p in shots] if shots else None,
                    }
                    rec = run_trial(client, trial, prompt, len(view["rooms"]), view, parse_retries, shots)
                    if rec["error"]:
                        n_api += 1
                        last_error = rec["error"]
                        time.sleep(0.5)
                    elif rec["parse_ok"]:
                        n_ok += 1
                    if not rec["parse_ok"]:
                        n_fail += 1
                    f.write(json.dumps({**base, **rec}, ensure_ascii=False) + "\n")
                    f.flush()
                    if n_ok == 0 and n_api >= FAIL_FAST_AFTER and n_api == i:
                        hint = ("Is Ollama running? Open the Ollama app (or run `ollama serve`) and check "
                                f"`ollama list` shows {cfg.model}." if cfg.provider == "ollama"
                                else "Check base_url, your API key in .env, and your internet connection.")
                        raise SystemExit(f"\nStopped: the first {n_api} calls to {cfg.name} all failed.\n"
                                         f"Last error: {last_error}\n{hint}")
                    if not quiet and (i % 25 == 0 or i == len(todo)):
                        msg = f"  {i}/{len(todo)} done | failures so far: {n_fail}"
                        if last_error:
                            msg += f" | last API error: {last_error[:120]}"
                        print(msg)
        finally:
            if shooter:
                shooter.__exit__(None, None, None)
            if site:
                site.__exit__(None, None, None)
        summary[cfg.name] = {"path": out, "ran": len(todo), "failures": n_fail, "api_errors": n_api}
    return summary


def hello(model_name: str) -> dict[str, Any]:
    """Week-1 check: can this model return a valid JSON answer?"""
    cfg = ModelConfig.load(model_name)
    client = ModelClient(cfg)
    prompt = ('Pick a number between 1 and 4. Respond ONLY with a JSON object in exactly this format: '
              '{"option_number": <1-4>, "hotel_name": "", "reason": "<one sentence>"}')
    try:
        comp = client.complete(prompt, seed=1, n_options=4)
    except Exception as e:
        hint = ("Is Ollama running and is the model pulled? Try `ollama list`." if cfg.provider == "ollama"
                else "Check base_url and the API key in .env.")
        return {"model": cfg.name, "parse_ok": False, "error": f"{type(e).__name__}: {e}", "hint": hint}
    parsed = parse_choice(comp.text, 4)
    return {"model": cfg.name, "raw_output": comp.text, "parse_ok": parsed.ok, "option": parsed.option_number,
            "note": parsed.error, "finish_reason": comp.finish_reason, "input_tokens": comp.input_tokens,
            "latency_s": round(comp.latency_s, 2), "version": client.describe_version()}
