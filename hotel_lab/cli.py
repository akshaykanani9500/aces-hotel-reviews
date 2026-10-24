"""Command line for Hotel Lab.   python -m hotel_lab <command> --help

Typical order:
  validate  -> checks -> plan -> serve (look at pages) -> hello (test a model)
  -> run (pilot with --limit) -> export -> analyze
  demo      -> runs the whole pipeline with a mock model (no AI model needed)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from hotel_lab import CONFIG_DIR, RUNS_DIR


def _load_env() -> None:
    """Read KEY=VALUE lines from .env in the repo root (for OPENROUTER_API_KEY, HF_TOKEN ...)."""
    from hotel_lab import REPO_ROOT
    import os
    p = REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.split(" #")[0].strip().strip('"').strip("'")
            if v and "PLACEHOLDER" not in v and not v.startswith("{"):   # ignore unfilled template values
                os.environ.setdefault(k.strip(), v)


def cmd_validate(args) -> int:
    from hotel_lab.design import load_config, used_levels
    from hotel_lab.stimuli import load_scenarios, validate_all
    cfg = load_config(args.config)
    scs = load_scenarios()
    fmt = used_levels(cfg, "format")
    summary_sets = used_levels(cfg, "review_set") if ({"summary", "both"} & set(fmt)) else []
    rep = validate_all(scs, style_levels=used_levels(cfg, "style"), summary_sets=summary_sets)
    for w in rep.warnings:
        print(f"WARNING  {w}")
    for e in rep.errors:
        print(f"ERROR    {e}")
    print(f"\n{len(scs)} scenario(s): {', '.join(s.id for s in scs)}")
    print("Validation passed." if rep.ok else f"Validation FAILED with {len(rep.errors)} error(s).")
    return 0 if rep.ok else 1


def cmd_checks(args) -> int:
    from hotel_lab.checks import run_checks
    from hotel_lab.stimuli import load_scenarios
    out = run_checks(load_scenarios())
    print(f"Manipulation-check files written to {out}")
    vs = out / "variant_summary.csv"
    if vs.exists():
        print(vs.read_text(encoding="utf-8"))
    return 0


def cmd_plan(args) -> int:
    from hotel_lab.design import build_plan, load_config, save_plan
    from hotel_lab.stimuli import load_scenarios
    if cmd_validate(args) != 0:
        print("Fix the validation errors before planning.")
        return 1
    cfg = load_config(args.config)
    trials = build_plan(cfg, load_scenarios())
    # Pre-flight: every trial must render (catches e.g. a style variant missing for a review set)
    from hotel_lab.conditions import build_view
    sc_map = {s.id: s for s in load_scenarios()}
    problems = {}
    for t in trials:
        try:
            build_view(sc_map[t["scenario_id"]], t)
        except (KeyError, ValueError) as e:
            problems.setdefault(str(e), t["trial_id"])
    if problems:
        for msg, tid in problems.items():
            print(f"ERROR    {msg}  (first seen in trial {tid})")
        print("Plan NOT saved. Fix the stimulus file or experiment.yaml.")
        return 1
    p = save_plan(trials, cfg["experiment_name"])
    cfg_path = args.config or CONFIG_DIR / "experiment.yaml"
    shutil.copy(cfg_path, p.parent / "experiment.yaml")
    from hotel_lab.design import manifest
    man = manifest(cfg_path, load_scenarios())
    (p.parent / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    if man.get("uncommitted_changes"):
        print("Note: hotel_lab has uncommitted changes. Commit before the main run so the manifest "
              "points to the exact code.")
    from collections import Counter
    by_block = Counter(t["block"] for t in trials)
    cells = Counter((t["block"], t["cell"]) for t in trials)
    print(f"\nPlan for '{cfg['experiment_name']}': {len(trials)} trials -> {p}")
    for b, n in by_block.items():
        n_cells = len([c for c in cells if c[0] == b])
        n_rt = sum(1 for t in trials if t["block"] == b and t.get("retest"))
        extra = f" incl. {n_rt} retest copies" if n_rt else ""
        print(f"  {b:<20} {n_cells} cells, {n} trials ({(n - n_rt) // max(n_cells, 1)} per cell{extra})")
    print(f"Calls per model and arm: {len(trials):,} (+ parse retries). "
          f"E.g. 5 models x text arm: ~{5 * len(trials):,} calls.")
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from hotel_lab.design import load_config
    from hotel_lab.site.app import create_app
    exp = args.experiment or load_config(args.config)["experiment_name"]
    print(f"StayCompare running at http://127.0.0.1:{args.port}  (experiment: {exp}). Press Ctrl+C to stop.")
    cfg = load_config(args.config)
    uvicorn.run(create_app(exp, cfg.get("show_category_scores", False)), host="127.0.0.1", port=args.port,
                log_level="warning")
    return 0


def cmd_hello(args) -> int:
    from hotel_lab.runner import hello
    res = hello(args.model)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["parse_ok"] else 1


def cmd_run(args) -> int:
    from hotel_lab.design import load_config
    from hotel_lab.runner import run
    cfg = load_config(args.config)
    exp = args.experiment or cfg["experiment_name"]
    res = run(exp, args.models, interface=args.interface or cfg.get("interface", "text"),
              reasoning_depth=args.reasoning or cfg.get("reasoning_depth", "low"),
              parse_retries=cfg.get("parse_retries", 1) if args.retries is None else args.retries,
              blocks=args.blocks, scenarios=args.scenarios, limit=args.limit, per_cell=args.per_cell,
              show_category_scores=cfg.get("show_category_scores", False))
    bad = False
    for name, r in res.items():
        print(f"{name}: ran {r['ran']} trials, {r['failures']} not usable ({r['api_errors']} API errors)")
        if r["ran"] and r["failures"] == r["ran"]:
            bad = True
    if bad:
        print("Every trial failed for at least one model. Look at `raw_output` / `error` in the log file.")
        return 1
    print("Done. Next:  python -m hotel_lab export  &&  python -m hotel_lab analyze")
    return 0


def cmd_export(args) -> int:
    from hotel_lab.design import load_config
    from hotel_lab.export import export
    exp = args.experiment or load_config(args.config)["experiment_name"]
    t, l = export(exp)
    print(f"Wrote {t}\nWrote {l}")
    return 0


def cmd_analyze(args) -> int:
    from hotel_lab.analysis import analyze
    from hotel_lab.design import load_config
    exp = args.experiment or load_config(args.config)["experiment_name"]
    out = analyze(exp, bootstrap_reps=args.bootstrap)
    print(f"Analysis written to {out}")
    import pandas as pd
    fr = pd.read_csv(out / "1_failure_rates.csv")
    bad = fr[fr["over_5pct"]]
    print(f"Cells with parse failures over 5%: {len(bad)} of {len(fr)}")
    acc = pd.read_csv(out / "2_accuracy_by_cell.csv")
    if not acc.empty:
        print("\nShare choosing the fact-sheet-best room:")
        print(acc.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return 0


def cmd_summarize(args) -> int:
    from hotel_lab.stimuli import load_scenarios
    from hotel_lab.summarize import summarize
    out = summarize(load_scenarios(), args.model, args.sets)
    print(f"Summaries written to {out}. Check them with two coders before using them.")
    return 0


def cmd_agreement(args) -> int:
    """Agreement between two coders in a filled coder sheet (columns starting with coder1 / coder2)."""
    import pandas as pd
    from hotel_lab.analysis import cohen_kappa
    df = pd.read_csv(args.file, encoding="utf-8-sig")
    c1 = [c for c in df.columns if c.lower().startswith("coder1")]
    c2 = [c for c in df.columns if c.lower().startswith("coder2")]
    if not c1 or not c2:
        print("Need columns starting with 'coder1' and 'coder2'.")
        return 1
    d = df[[c1[0], c2[0]]].dropna()
    a = d[c1[0]].astype(str).str.strip().str.upper().tolist()
    b = d[c2[0]].astype(str).str.strip().str.upper().tolist()
    if not a:
        print("No rows coded by both coders yet.")
        return 1
    agree = sum(x == y for x, y in zip(a, b)) / len(a)
    print(f"Rows coded by both: {len(a)}\nPercent agreement: {agree:.1%}\nCohen's kappa: {cohen_kappa(a, b):.2f}")
    print("Brief: at least 20 items. Rows where coders disagree:")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"  row {d.index[i] + 2}: coder1={x} coder2={y}")
    return 0


def cmd_models(args) -> int:
    from hotel_lab.models import MODELS_DIR, ModelConfig
    for p in sorted(MODELS_DIR.glob("*.yaml")):
        c = ModelConfig.load(str(p))
        print(f"{c.name:<28} {c.provider:<18} {c.model:<38} vision={c.supports_vision}")
    return 0


def cmd_demo(args) -> int:
    """End-to-end smoke test with the mock model: plan -> run -> export -> analyze."""
    from hotel_lab.analysis import analyze
    from hotel_lab.design import build_plan, load_config, save_plan
    from hotel_lab.export import export
    from hotel_lab.runner import run
    from hotel_lab.stimuli import load_scenarios
    if cmd_validate(args) != 0:
        return 1
    cfg = load_config(args.config)
    cfg = {**cfg, "experiment_name": "demo_mock", "repeats": 2}
    out_dir = RUNS_DIR / "demo_mock"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    trials = build_plan(cfg, load_scenarios())
    p = save_plan(trials, "demo_mock")
    import yaml
    with open(p.parent / "experiment.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    run("demo_mock", ["mock", "mock-biased"], interface="text", quiet=False)
    export("demo_mock")
    out = analyze("demo_mock", bootstrap_reps=200)
    import pandas as pd
    pos = pd.read_csv(out / "4_position_shares.csv")
    print("\nPosition shares (mock picks at random; mock-biased prefers option 1):")
    print(pos.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nDemo finished. Look at {out}")
    return 0


def main(argv=None) -> None:
    # Windows consoles may not be UTF-8; never crash on printing ₹ or emoji
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    _load_env()
    ap = argparse.ArgumentParser(prog="python -m hotel_lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="experiment YAML (default: hotel_lab/config/experiment.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="check the stimulus files")
    sub.add_parser("checks", help="manipulation checks + coder sheets")
    sub.add_parser("plan", help="build the trial plan from experiment.yaml")
    s = sub.add_parser("serve", help="open the mock hotel website locally")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--experiment", default=None)
    s = sub.add_parser("hello", help="week-1 test: does a model return valid JSON?")
    s.add_argument("--model", required=True)
    s = sub.add_parser("run", help="run planned trials")
    s.add_argument("--models", nargs="+", required=True, help="model config names, e.g. ollama-qwen2.5-7b")
    s.add_argument("--experiment", default=None)
    s.add_argument("--interface", choices=["text", "screenshot"], default=None)
    s.add_argument("--reasoning", choices=["low", "high"], default=None)
    s.add_argument("--retries", type=int, default=None)
    s.add_argument("--blocks", nargs="+", default=None)
    s.add_argument("--scenarios", nargs="+", default=None)
    s.add_argument("--per-cell", type=int, default=None, help="pilot: the first N trials of EVERY cell")
    s.add_argument("--limit", type=int, default=None, help="at most N trials, spread across all cells")
    s = sub.add_parser("export", help="raw logs -> tidy CSV tables")
    s.add_argument("--experiment", default=None)
    s = sub.add_parser("analyze", help="shares, effects, kappa, conditional logit, figures")
    s.add_argument("--experiment", default=None)
    s.add_argument("--bootstrap", type=int, default=1000)
    s = sub.add_parser("summarize", help="generate review summaries with a fixed prompt")
    s.add_argument("--model", required=True)
    s.add_argument("--sets", nargs="+", default=None)
    s = sub.add_parser("agreement", help="coder agreement for a filled coder sheet CSV")
    s.add_argument("file")
    sub.add_parser("models", help="list model configs")
    sub.add_parser("demo", help="whole pipeline with a mock model (no AI needed)")

    args = ap.parse_args(argv)
    handler = {"validate": cmd_validate, "checks": cmd_checks, "plan": cmd_plan, "serve": cmd_serve,
               "hello": cmd_hello, "run": cmd_run, "export": cmd_export, "analyze": cmd_analyze,
               "summarize": cmd_summarize, "models": cmd_models, "agreement": cmd_agreement, "demo": cmd_demo}[args.cmd]
    sys.exit(handler(args))
