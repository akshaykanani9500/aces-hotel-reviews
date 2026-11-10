"""Hotel Lab Studio: the local control panel for the whole study.

    python -m hotel_lab studio        ->  http://127.0.0.1:8800

Left:   pick a hypothesis (from the Sheet's Hypotheses tab); see H0/Ha and what is held constant.
Centre: see exactly what the agent sees (the hotel pages, the JSON, the prompt) and re-randomise.
Right:  add / test AI models (local Ollama, OpenAI, Gemini, Anthropic, OpenRouter, Groq, HF, ...).
Bottom: run, watch progress, analyse, read verdicts. Results go to the local log AND the Google Sheet.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hotel_lab import REPO_ROOT, RUNS_DIR, sheetdb, study
from hotel_lab.models import MODELS_DIR, ModelConfig

STATIC = Path(__file__).resolve().parent / "static"
ENV_FILE = REPO_ROOT / ".env"

PROVIDER_PRESETS = {
    "ollama": {"label": "Ollama (local, free)", "provider": "ollama", "base_url": "http://localhost:11434",
               "api_key_env": "", "example": "qwen2.5:7b", "serving": "ollama"},
    "lmstudio": {"label": "LM Studio (local, free)", "provider": "openai_compatible", "base_url": "http://localhost:1234/v1",
                 "api_key_env": "", "example": "qwen2.5-7b-instruct", "serving": "lmstudio"},
    "openrouter": {"label": "OpenRouter (many open models)", "provider": "openai_compatible",
                   "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
                   "example": "meta-llama/llama-3.3-70b-instruct", "serving": "openrouter"},
    "groq": {"label": "Groq (fast open models)", "provider": "openai_compatible", "base_url": "https://api.groq.com/openai/v1",
             "api_key_env": "GROQ_API_KEY", "example": "llama-3.1-8b-instant", "serving": "groq"},
    "hf": {"label": "Hugging Face router", "provider": "openai_compatible", "base_url": "https://router.huggingface.co/v1",
           "api_key_env": "HF_TOKEN", "example": "Qwen/Qwen2.5-72B-Instruct", "serving": "hf"},
    "openai": {"label": "OpenAI (closed)", "provider": "openai_compatible", "base_url": "https://api.openai.com/v1",
               "api_key_env": "OPENAI_API_KEY", "example": "gpt-4.1-mini", "serving": "openai"},
    "gemini": {"label": "Google Gemini (closed)", "provider": "openai_compatible",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key_env": "GEMINI_API_KEY",
               "example": "gemini-2.5-flash", "serving": "gemini"},
    "anthropic": {"label": "Anthropic Claude (closed)", "provider": "anthropic", "base_url": "https://api.anthropic.com",
                  "api_key_env": "ANTHROPIC_API_KEY", "example": "claude-haiku-4-5", "serving": "anthropic"},
    "custom": {"label": "Other OpenAI-compatible server", "provider": "openai_compatible", "base_url": "http://localhost:8000/v1",
               "api_key_env": "", "example": "", "serving": "custom"},
    "mock": {"label": "Test agent (no AI, for checking the pipeline)", "provider": "mock", "base_url": "",
             "api_key_env": "", "example": "mock-simulated", "serving": "mock"},
}


# ----------------------------------------------------------------------------- .env helpers
def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v and "PLACEHOLDER" not in v and not v.startswith("{"):
                os.environ.setdefault(k.strip(), v)


def save_env_key(name: str, value: str) -> None:
    """Store an API key in .env (git-ignored) and in this process. Keys never go to the Sheet."""
    if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,60}", name):
        raise HTTPException(400, "Key name must look like OPENAI_API_KEY")
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [l for l in lines if not l.strip().startswith(name + "=")]
    lines.append(f'{name}="{value}"')
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[name] = value


# ----------------------------------------------------------------------------- job runner
class Job:
    def __init__(self, hypothesis_id: str, block: str, models: list[str], mode: str):
        self.id = study.new_run_id()
        self.hypothesis_id, self.block, self.models, self.mode = hypothesis_id, block, models, mode
        self.status = "starting"
        self.done = self.total = self.failures = 0
        self.current_model = ""
        self.last_error = ""
        self.log: list[str] = []
        self.started = time.time()
        self.finished: Optional[float] = None
        self.stop = False
        self.pushed = 0
        self.planned = 0
        self.experiment = ""

    def say(self, msg: str) -> None:
        self.log.append(time.strftime("%H:%M:%S ") + msg)
        self.log = self.log[-200:]

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "stop"}
        d["elapsed_s"] = round((self.finished or time.time()) - self.started, 1)
        return d


JOB: dict[str, Optional[Job]] = {"current": None}


def _run_job(job: Job, per_cell: Optional[int], interface: str, reasoning: str) -> None:
    from hotel_lab.runner import run
    try:
        st = study.load_study()
        info = study.prepare(job.block, st)
        job.experiment = info["experiment"]
        push = str(st["settings"].get("push_results", "TRUE")).upper() == "TRUE"
        pusher = study.SheetPusher(info["experiment"], job.id, st["source"].get("sha"), enabled=push)
        if push and pusher.db is None:
            job.say("No Sheet linked: results are saved locally and will be pushed later (outbox).")
        job.say(f"Plan ready: {info['trials']} trials ({info['per_cell']} per cell). Mode: {job.mode}.")
        job.status = "running"
        for m in job.models:
            if job.stop:
                break
            job.current_model = m
            job.say(f"Running {m} ...")

            def on_record(rec: dict[str, Any], _m=m) -> None:
                job.done += 1
                job.total = max(job.total, rec.get("_total", 0))
                if not rec.get("parse_ok"):
                    job.failures += 1
                if rec.get("error"):
                    job.last_error = str(rec["error"])[:300]
                pusher.add(rec)
                job.pushed = pusher.pushed

            try:
                res = run(info["experiment"], [m], interface=interface, reasoning_depth=reasoning,
                          parse_retries=int(st["settings"].get("parse_retries", 1)), per_cell=per_cell,
                          quiet=True, on_record=on_record, should_stop=lambda: job.stop)
                r = res.get(m) or next(iter(res.values()), {})
                job.planned += r.get("planned", 0)
                job.say(f"{m}: {r.get('ran', 0)} of {r.get('planned', 0)} trials run, {r.get('failures', 0)} not usable.")
            except SystemExit as e:          # e.g. model server not reachable (fail-fast)
                job.say(f"{m} stopped: {e}")
                job.last_error = str(e)
            pusher.flush()
            job.pushed = pusher.pushed
        if pusher.last_error:
            job.say(f"Some rows could not be sent to the Sheet ({pusher.last_error}); they wait in the outbox.")
        job.status = "stopped" if job.stop else "finished"
        job.say("Done. Press 'Analyse all' to update the verdicts.")
    except Exception as e:
        job.status = "error"
        job.last_error = f"{type(e).__name__}: {e}"
        job.say(job.last_error)
        job.say(traceback.format_exc()[-800:])
    finally:
        job.finished = time.time()
        db = sheetdb.get_db()
        if db is not None and job.experiment:
            commit = ""
            try:
                import subprocess
                commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                                        text=True, timeout=5).stdout.strip()
            except Exception:
                commit = "(git not found)"
            try:
                db.append("Runs", [{"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": job.id,
                                    "experiment": job.experiment, "block": job.block, "models": ", ".join(job.models),
                                    "trials_planned": job.planned, "trials_done": job.done, "failures": job.failures,
                                    "snapshot_sha": (sheetdb.current_snapshot() or {}).get("sha", "yaml"),
                                    "git_commit": commit, "status": job.status, "notes": job.mode}])
            except Exception as e:
                job.say(f"Could not write the Runs tab: {e}")


# ----------------------------------------------------------------------------- request models
class SourceIn(BaseModel):
    sheet_url: str
    sheet_token: str


class HypIn(BaseModel):
    hypothesis_id: str


class PreviewIn(BaseModel):
    hypothesis_id: str
    cell: Optional[str] = None
    seed: Optional[int] = None


class ModelIn(BaseModel):
    name: str
    preset: str = "ollama"
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 400
    num_ctx: int = 16384
    supports_vision: bool = False
    json_mode: bool = False
    family: str = ""
    size_b: Optional[float] = None


class RunIn(BaseModel):
    hypothesis_id: str
    models: list[str]
    mode: str = "pilot"           # pilot | full
    per_cell: int = 5
    interface: str = "text"
    reasoning: str = "low"


# ----------------------------------------------------------------------------- app
def create_studio(site_port: Optional[int] = None) -> FastAPI:
    load_env()
    app = FastAPI(title="Hotel Lab Studio")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="studio_static")

    from hotel_lab.screenshots import LocalSite
    try:
        site = LocalSite(experiment=None, port=site_port).__enter__()   # the hotel website the agents see
    except Exception as e:
        raise SystemExit(f"Could not start the hotel website on port {site_port}: {e}\n"
                         f"Another 'python -m hotel_lab serve' or Studio is probably still running. Close it, or start "
                         f"with a different port:  python -m hotel_lab studio --site-port 8790")

    def hyp(hid: str, st: dict[str, Any]) -> dict[str, Any]:
        for h in st["hypotheses"]:
            if h["hypothesis_id"] == hid:
                return h
        raise HTTPException(404, f"Unknown hypothesis {hid}")

    def models_list() -> list[dict[str, Any]]:
        out = []
        for p in sorted(MODELS_DIR.glob("*.yaml")):
            try:
                c = ModelConfig.load(str(p))
            except Exception as e:
                out.append({"name": p.stem, "error": str(e)})
                continue
            needs = c.api_key_env or ("ANTHROPIC_API_KEY" if c.provider == "anthropic" else "")
            out.append({"name": c.name, "provider": c.provider, "model": c.model, "base_url": c.base_url,
                        "serving": c.serving, "vision": c.supports_vision, "temperature": c.temperature,
                        "key_env": needs, "key_set": (not needs) or bool(os.getenv(needs)),
                        "local": p.stem.startswith("local-")})
        return out

    def held_constant(h: dict[str, Any], st: dict[str, Any]) -> list[str]:
        s = st["settings"]
        if "facts_only" in (h.get("levels") or ""):
            return ["NO reviews are shown: each room shows only its 3 facts as plain labels (Good/Average/Poor)",
                    "price, amenities, room type, guest rating, review count and photo: identical for all rooms",
                    "room positions balanced; hotel names rotate"]
        manipulated = {h.get("factor"), h.get("factor2")}
        fixed = {"order": s.get("default_order", "random"), "style": s.get("default_style", "base"),
                 "review_set": s.get("default_review_set", "base"), "format": s.get("default_format", "full")}
        lines = [f"{k} = {v}" for k, v in fixed.items() if k not in manipulated]
        lines += ["price, amenities, room type, guest rating, review count and photo: identical for all rooms",
                  "room positions: balanced (each room in each position equally often)",
                  "hotel names: rotate across rooms", "hidden facts: never shown to the agent"]
        return lines

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/state")
    def state():
        st = study.load_study()
        local = sheetdb.load_local_settings()
        bl = study.blocks(st["hypotheses"])
        name = st["settings"].get("experiment_name", "study_v1")
        block_status = {}
        for b in bl:
            raw = RUNS_DIR / f"{name}__{b}" / "raw"
            counts = {}
            if raw.exists():
                for f in raw.glob("*.jsonl"):
                    counts[f.stem] = sum(1 for _ in open(f, encoding="utf-8"))
            block_status[b] = counts
        hyps = []
        for h in st["hypotheses"]:
            hyps.append({**h, "held_constant": held_constant(h, st)})
        return {
            "source": st["source"], "sheet_linked": bool(local.get("sheet_url")),
            "sheet_url_hint": (local.get("sheet_url", "")[:60] + "...") if local.get("sheet_url") else "",
            "settings": st["settings"], "hypotheses": hyps, "scenarios": [s.id for s in st["scenarios"]],
            "block_status": block_status, "models": models_list(), "presets": PROVIDER_PRESETS,
            "site_url": site.base_url, "job": JOB["current"].as_dict() if JOB["current"] else None,
        }

    # ---- data source ----
    @app.post("/api/source")
    def set_source(body: SourceIn):
        db = sheetdb.AppsScriptDB(body.sheet_url, body.sheet_token)
        try:
            tabs = db.read_all()
        except Exception as e:
            raise HTTPException(400, f"Could not read the Sheet: {e}")
        sheetdb.save_local_settings(sheet_url=body.sheet_url.strip(), sheet_token=body.sheet_token.strip())
        return {"ok": True, "tabs": {k: len(v) for k, v in tabs.items()}}

    @app.post("/api/sync")
    def sync():
        try:
            return sheetdb.sync_from_sheet()
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post("/api/sync_xlsx")
    async def sync_xlsx(file: UploadFile = File(...)):
        tmp = RUNS_DIR / "_db" / "upload.xlsx"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(await file.read())
        try:
            return sheetdb.sync_from_xlsx(tmp)
        except Exception as e:
            raise HTTPException(400, f"Could not read that file: {e}")

    @app.post("/api/use_yaml")
    def use_yaml():
        if sheetdb.CURRENT.exists():
            sheetdb.CURRENT.unlink()
        return {"ok": True}

    @app.get("/api/workbook")
    def workbook():
        from hotel_lab.stimuli import STIMULI_DIR, load_scenarios
        p = RUNS_DIR / "_db" / "hotel_lab_database.xlsx"
        models = [{"name": m["name"], "provider": m.get("provider", ""), "base_url": m.get("base_url", ""),
                   "model": m.get("model", ""), "api_key_env": m.get("key_env", "")} for m in models_list()]
        sheetdb.build_workbook(p, load_scenarios(STIMULI_DIR, source="yaml"), models)
        return FileResponse(p, filename="hotel_lab_database.xlsx")

    # ---- experiment ----
    @app.post("/api/prepare")
    def prepare(body: HypIn):
        st = study.load_study()
        h = hyp(body.hypothesis_id, st)
        try:
            info = study.prepare(h["block"], st)
        except Exception as e:
            raise HTTPException(400, str(e))
        return info

    @app.post("/api/preview")
    def preview(body: PreviewIn):
        from hotel_lab.conditions import build_view
        from hotel_lab.design import load_plan, plan_path
        from hotel_lab.prompts import build_prompt
        st = study.load_study()
        h = hyp(body.hypothesis_id, st)
        exp = f"{st['settings'].get('experiment_name', 'study_v1')}__{h['block']}"
        if not plan_path(exp).exists():
            study.prepare(h["block"], st)
        trials = [t for t in load_plan(exp) if not t.get("retest")]
        cells = sorted({t["cell"] for t in trials})
        cell = body.cell if body.cell in cells else cells[0]
        pool = [t for t in trials if t["cell"] == cell]
        rng = random.Random(body.seed if body.seed is not None else time.time_ns())
        t = rng.choice(pool)
        sc = {s.id: s for s in st["scenarios"]}[t["scenario_id"]]
        view = build_view(sc, t)
        rooms = []
        for r in view["rooms"]:
            room = sc.room(r["room_id"])
            shown = r.get("verified_facts")
            facts = ({ln.split(": ")[0]: ln.split(": ")[-1] + " (shown)" for ln in shown} if shown
                     else {f["attribute"]: f["level"] for f in room["facts"]})
            rooms.append({"option": r["option_number"], "hotel_name": r["hotel_name"], "profile": r["room_id"],
                          "focal": r["room_id"] == t["focal_room_id"],
                          "best": (r["room_id"] == t["focal_room_id"]) if r["levels"].get("facts") in ("all_good", "all_poor")
                          else r["room_id"] == t["best_room_id"],
                          "levels": r["levels"], "facts": facts,
                          "first_review_stars": (r.get("reviews") or [{}])[0].get("rating")})
        return {"experiment": exp, "trial_id": t["trial_id"], "cell": cell, "cells": cells,
                "scenario": t["scenario_id"], "query": view["query"],
                "page_url": f"{site.base_url}/e/{exp}/trial/{t['trial_id']}",
                "json_url": f"{site.base_url}/e/{exp}/api/trial/{t['trial_id']}",
                "prompt": build_prompt(view, t["prompt_id"], "text", st["settings"].get("reasoning_depth", "low")),
                "rooms": rooms}

    # ---- models ----
    @app.post("/api/models")
    def save_model(body: ModelIn):
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,60}", body.name):
            raise HTTPException(400, "Short name: letters, numbers, - _ . only")
        if not body.model.strip() and body.preset != "mock":
            raise HTTPException(400, "Model name is empty: type the model id, e.g. qwen2.5:7b")
        pre = PROVIDER_PRESETS.get(body.preset)
        if pre is None:
            raise HTTPException(400, "Unknown provider")
        key_env = (body.api_key_env or pre["api_key_env"] or "").strip()
        if body.api_key:
            key_env = key_env or f"HOTEL_LAB_KEY_{re.sub(r'[^A-Z0-9]', '_', body.name.upper())}"
            save_env_key(key_env, body.api_key.strip())
        cfg = {"provider": pre["provider"], "base_url": body.base_url or pre["base_url"], "model": body.model,
               "family": body.family, "size_b": body.size_b, "serving": pre["serving"],
               "temperature": body.temperature, "max_tokens": body.max_tokens,
               "supports_vision": body.supports_vision, "json_mode": body.json_mode}
        if pre["provider"] == "ollama":
            cfg["num_ctx"] = body.num_ctx
        if key_env:
            cfg["api_key_env"] = key_env
        if pre["provider"] == "mock":
            cfg = {"provider": "mock", "model": body.model or "mock-random", "serving": "mock", "family": "mock",
                   "strategy": "simulated" if "simulated" in (body.model or "") else "random"}
        path = MODELS_DIR / f"local-{body.name}.yaml"
        path.write_text("# Added from Hotel Lab Studio. API keys are NOT stored here (see .env).\n" +
                        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return {"ok": True, "name": path.stem}

    @app.delete("/api/models/{name}")
    def delete_model(name: str):
        p = MODELS_DIR / f"{name}.yaml"
        if not name.startswith("local-") or not p.exists():
            raise HTTPException(400, "Only models added from the Studio (local-...) can be deleted here")
        p.unlink()
        return {"ok": True}

    @app.post("/api/models/{name}/test")
    def test_model(name: str):
        from hotel_lab.runner import hello
        try:
            return hello(name)
        except Exception as e:
            return {"parse_ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---- runs ----
    @app.post("/api/run")
    def start_run(body: RunIn):
        cur = JOB["current"]
        if cur and cur.status in ("starting", "running"):
            raise HTTPException(409, "A run is already in progress")
        if not body.models:
            raise HTTPException(400, "Tick at least one model on the right")
        st = study.load_study()
        h = hyp(body.hypothesis_id, st)
        job = Job(body.hypothesis_id, h["block"], body.models, body.mode)
        JOB["current"] = job
        per_cell = body.per_cell if body.mode == "pilot" else None
        threading.Thread(target=_run_job, args=(job, per_cell, body.interface, body.reasoning), daemon=True).start()
        return job.as_dict()

    @app.post("/api/stop")
    def stop():
        if JOB["current"]:
            JOB["current"].stop = True
        return {"ok": True}

    @app.get("/api/job")
    def job():
        return JOB["current"].as_dict() if JOB["current"] else {}

    @app.get("/api/plan_info")
    def plan_info(hypothesis_id: str):
        """Size of the full plan for a hypothesis, without saving anything."""
        from hotel_lab.design import build_plan
        st = study.load_study()
        h = hyp(hypothesis_id, st)
        cfg = study.block_config(h["block"], st)
        trials = build_plan(cfg, st["scenarios"])
        main = [t for t in trials if not t.get("retest")]
        cells = sorted({t["cell"] for t in main})
        return {"block": h["block"], "cells": cells, "per_cell": len(main) // max(len(cells), 1),
                "trials": len(trials), "retest_copies": len(trials) - len(main),
                "scenarios": len(st["scenarios"]), "shared_with": [x["hypothesis_id"] for x in st["hypotheses"]
                                                                  if x.get("block") == h["block"] and x["hypothesis_id"] != hypothesis_id]}

    @app.get("/api/verdicts")
    def last_verdicts():
        """The verdict table from the last 'Analyse', if any."""
        import pandas as pd
        st = study.load_study()
        p = RUNS_DIR / f"{st['settings'].get('experiment_name', 'study_v1')}__verdicts" / "verdicts.csv"
        if not p.exists():
            return {"rows": [], "csv": None}
        df = pd.read_csv(p, keep_default_na=False, na_values=[""])
        rows = [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in df.to_dict("records")]
        return {"rows": rows, "csv": str(p), "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))}

    @app.post("/api/analyze")
    def analyze():
        res = study.analyze()
        rows = [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()} for r in res["rows"]]
        return {"rows": rows, "csv": res["csv"], "pushed": res["pushed"]}

    @app.post("/api/push_pending")
    def push_pending():
        st = study.load_study()
        name = st["settings"].get("experiment_name", "study_v1")
        out = {}
        for b in study.blocks(st["hypotheses"]):
            try:
                out[b] = study.push_pending(f"{name}__{b}")
            except Exception as e:
                out[b] = {"error": str(e)}
        return out

    @app.on_event("shutdown")
    def _shutdown():
        site.__exit__(None, None, None)

    return app
