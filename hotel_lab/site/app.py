"""StayCompare: the local mock hotel website (FastAPI + Jinja templates).

Adapted from the ACES sandbox (sandbox/app.py), which shows a product search grid.
Here every trial has a results page and one page per room with its reviews.

Run:  python -m hotel_lab serve        then open http://127.0.0.1:8765
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hotel_lab import PACKAGE_DIR
from hotel_lab.conditions import build_view, preview_trial
from hotel_lab.design import ORDER_LEVELS, load_plan
from hotel_lab.prompts import listings_json
from hotel_lab.stimuli import load_scenarios

SITE_DIR = PACKAGE_DIR / "site"
CURRENCY_SYMBOL = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


def money(value: float, currency: str = "INR") -> str:
    v = int(round(float(value)))
    if currency == "INR":  # Indian digit grouping: 1,23,456
        s = str(abs(v))
        if len(s) > 3:
            head, tail = s[:-3], s[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            if head:
                groups.insert(0, head)
            s = ",".join(groups) + "," + tail
        return f"₹{'-' if v < 0 else ''}{s}"
    return f"{CURRENCY_SYMBOL.get(currency, currency + ' ')}{v:,}"


class PreviewError(Exception):
    pass


def create_app(experiment: Optional[str] = None, show_category_scores: bool = False) -> FastAPI:
    app = FastAPI(title="StayCompare (Hotel Lab sandbox)")

    @app.exception_handler(PreviewError)
    def _preview_error(request: Request, exc: PreviewError):
        return HTMLResponse(
            "<html><body style='font-family:Arial;padding:30px'><h2>This combination is not available</h2>"
            f"<p>{exc}</p><p>Most style variants exist only for the <b>base</b> review set. "
            "<a href='/'>Back</a></p></body></html>", status_code=400)
    app.mount("/static", StaticFiles(directory=str(SITE_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(SITE_DIR / "templates"))
    templates.env.globals["money"] = money

    scenarios = {s.id: s for s in load_scenarios()}
    trials: dict[str, dict[str, Any]] = {}
    if experiment:
        try:
            trials = {t["trial_id"]: t for t in load_plan(experiment)}
        except FileNotFoundError:
            trials = {}

    def _trial(trial_id: str) -> dict[str, Any]:
        if trial_id not in trials:
            raise HTTPException(404, f"Unknown trial {trial_id} in experiment {experiment}")
        return trials[trial_id]

    def _view(trial: dict[str, Any]) -> dict[str, Any]:
        sc = scenarios.get(trial["scenario_id"])
        if sc is None:
            raise HTTPException(404, f"Unknown scenario {trial['scenario_id']}")
        return build_view(sc, trial, show_category_scores)

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        first = list(trials.values())[:60]
        return templates.TemplateResponse(request, "home.html", {
            "view": None, "scenarios": list(scenarios.values()), "order_levels": sorted(ORDER_LEVELS),
            "experiment": experiment, "trials": first, "n_trials": len(trials)})

    # ---- experiment trials (what the agent sees) ----
    @app.get("/trial/{trial_id}", response_class=HTMLResponse)
    def trial_results(request: Request, trial_id: str):
        view = _view(_trial(trial_id))
        return templates.TemplateResponse(request, "results.html", {
            "view": view, "room_url": lambda r: f"/trial/{trial_id}/room/{r['room_id']}"})

    @app.get("/trial/{trial_id}/room/{room_id}", response_class=HTMLResponse)
    def trial_room(request: Request, trial_id: str, room_id: str):
        view = _view(_trial(trial_id))
        room = next((r for r in view["rooms"] if r["room_id"] == room_id), None)
        if room is None:
            raise HTTPException(404, "Unknown room")
        return templates.TemplateResponse(request, "room.html", {
            "view": view, "r": room, "back_url": f"/trial/{trial_id}"})

    @app.get("/api/trial/{trial_id}")
    def trial_json(trial_id: str):
        """The exact listing JSON a text-mode agent receives (hidden facts are never included)."""
        import json
        return JSONResponse(json.loads(listings_json(_view(_trial(trial_id)))))

    # ---- previews (for designing stimuli) ----
    def _preview(scenario: str, order: str, style: str, review_set: str, format: str, seed: int):
        sc = scenarios.get(scenario)
        if sc is None:
            raise HTTPException(404, f"Unknown scenario {scenario}")
        levels = {"order": order, "style": style, "review_set": review_set, "format": format}
        try:
            return build_view(sc, preview_trial(sc, levels, seed), show_category_scores)
        except (KeyError, ValueError) as e:
            raise PreviewError(str(e))

    @app.get("/preview", response_class=HTMLResponse)
    def preview(request: Request, scenario: str, order: str = "newest", style: str = "base",
                review_set: str = "base", format: str = "full", seed: int = 0):
        view = _preview(scenario, order, style, review_set, format, seed)
        q = urlencode({"order": order, "style": style, "review_set": review_set, "format": format, "seed": seed})
        return templates.TemplateResponse(request, "results.html", {
            "view": view, "room_url": lambda r: f"/preview/{scenario}/room/{r['room_id']}?{q}"})

    @app.get("/preview/{scenario}/room/{room_id}", response_class=HTMLResponse)
    def preview_room(request: Request, scenario: str, room_id: str, order: str = "newest", style: str = "base",
                     review_set: str = "base", format: str = "full", seed: int = 0):
        view = _preview(scenario, order, style, review_set, format, seed)
        room = next((r for r in view["rooms"] if r["room_id"] == room_id), None)
        if room is None:
            raise HTTPException(404, "Unknown room")
        q = urlencode({"scenario": scenario, "order": order, "style": style, "review_set": review_set,
                       "format": format, "seed": seed})
        return templates.TemplateResponse(request, "room.html", {"view": view, "r": room, "back_url": f"/preview?{q}"})

    return app
