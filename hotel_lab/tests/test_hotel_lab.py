from pathlib import Path
"""Tests for Hotel Lab.  Run from the repo root:  python -m pytest hotel_lab"""
import json

import pytest
from fastapi.testclient import TestClient

from hotel_lab.conditions import build_view, order_reviews, preview_trial
from hotel_lab.design import build_plan, load_config, stable_seed
from hotel_lab.parse import parse_choice
from hotel_lab.prompts import build_prompt, listings_json
from hotel_lab.site.app import create_app, money
from hotel_lab.stimuli import load_scenarios, validate_all
from hotel_lab.design import used_levels


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_stimuli_validate(scenarios, cfg):
    rep = validate_all(scenarios, used_levels(cfg, "style"), used_levels(cfg, "review_set"))
    assert rep.ok, rep.errors


def test_plan_is_reproducible(scenarios, cfg):
    a, b = build_plan(cfg, scenarios), build_plan(cfg, scenarios)
    assert a == b
    assert len({t["trial_id"] for t in a}) == len(a)


def test_positions_are_shuffled(scenarios, cfg):
    plan = build_plan(cfg, scenarios)
    firsts = {t["room_order"][0] for t in plan}
    assert len(firsts) == 4


def test_focal_assignment(scenarios, cfg):
    plan = [t for t in build_plan(cfg, scenarios) if t["block"] == "style_emoji" and t["cell"] == "style=emoji"]
    for t in plan:
        for rid, lv in t["room_levels"].items():
            assert lv["style"] == ("emoji" if rid == t["focal_room_id"] else "base")
            assert lv["order"] == "random" and lv["review_set"] == "base" and lv["format"] == "full"


def test_view_never_leaks_facts(scenarios, cfg):
    t = build_plan(cfg, scenarios)[0]
    sc = next(s for s in scenarios if s.id == t["scenario_id"])
    txt = listings_json(build_view(sc, t))
    for room in sc.rooms:
        for f in room["facts"]:
            assert f["statement"] not in txt
    assert '"levels"' not in txt and '"room_id"' not in txt
    assert "best_room" not in txt


def test_order_changes_order_not_content(scenarios):
    room = scenarios[0].rooms[0]
    pool = [scenarios[0].review(room, i) for i in room["review_sets"]["base"]]
    import random
    for lvl in ["newest", "most_helpful", "random", "extreme_first", "best_first", "worst_first", "oldest"]:
        out = order_reviews(pool, lvl, random.Random(1))
        assert sorted(r["review_id"] for r in out) == sorted(r["review_id"] for r in pool)
    newest = order_reviews(pool, "newest", random.Random(1))
    assert newest[0]["posted_date"] == max(r["posted_date"] for r in pool)


def test_same_stars_across_styles(scenarios):
    sc = scenarios[0]
    v1 = build_view(sc, preview_trial(sc, {"order": "newest", "style": "base", "review_set": "base", "format": "full"}))
    v2 = build_view(sc, preview_trial(sc, {"order": "newest", "style": "emoji", "review_set": "base", "format": "full"}))
    for a, b in zip(v1["rooms"], v2["rooms"]):
        assert [r["rating"] for r in a["reviews"]] == [r["rating"] for r in b["reviews"]]
        assert a["guest_rating"] == b["guest_rating"] and a["price_per_night"] == b["price_per_night"]
        assert [r["text"] for r in a["reviews"]] != [r["text"] for r in b["reviews"]]


def test_summary_format_hides_reviews(scenarios):
    sc = scenarios[0]
    v = build_view(sc, preview_trial(sc, {"order": "newest", "style": "base", "review_set": "base", "format": "summary"}))
    assert all("reviews" not in r and r["review_summary"] for r in v["rooms"])


@pytest.mark.parametrize("text,expected", [
    ('{"option_number": 3, "hotel_name": "X", "reason": "y"}', 3),
    ('Sure! ```json\n{"option_number": "2", "reason": "ok"}\n```', 2),
    ('Thinking... {"note": 1} final: {"option_number": 4}', 4),
    ('{"option_number": 9}', None),
    ("I like option 2", None),
    ("", None),
])
def test_parse(text, expected):
    assert parse_choice(text, 4).option_number == expected


def test_parse_failure_is_not_a_choice():
    p = parse_choice("no idea", 4)
    assert not p.ok and p.option_number is None


def test_prompt_contains_all_options(scenarios, cfg):
    t = build_plan(cfg, scenarios)[0]
    sc = next(s for s in scenarios if s.id == t["scenario_id"])
    p = build_prompt(build_view(sc, t), "main", "text", "low")
    assert '"option_number": 4' in p and "{" in p and "{listings}" not in p


def test_money():
    assert money(1234567, "INR") == "₹12,34,567"
    assert money(9100, "INR") == "₹9,100"
    assert money(950, "INR") == "₹950"


def test_site_pages(scenarios):
    c = TestClient(create_app(None))
    assert c.get("/").status_code == 200
    sid = scenarios[0].id
    r = c.get(f"/preview?scenario={sid}&format=both")
    assert r.status_code == 200 and "Option 1" in r.text
    r = c.get(f"/preview/{sid}/room/r1?format=both&style=emoji")
    assert r.status_code == 200 and "Review summary" in r.text and "Guest reviews" in r.text
    assert c.get(f"/preview?scenario={sid}&style=nonexistent").status_code == 400


def test_stable_seed():
    assert stable_seed(1, "a") == stable_seed(1, "a") != stable_seed(1, "b")


def test_layout_shared_across_cells_and_latin_balanced(scenarios, cfg):
    plan = [t for t in build_plan(cfg, scenarios) if t["block"] == "format" and not t["retest"]]
    by_key = {}
    for t in plan:
        by_key.setdefault((t["scenario_id"], t["item"], t["repeat"]), set()).add(tuple(t["room_order"]))
    assert all(len(v) == 1 for v in by_key.values())  # same layout in full and summary cells
    # Latin square: over repeats 0-3 each room appears once in position 1
    t0 = [t for t in plan if t["cell"] == "format=full" and t["item"] == 0 and t["scenario_id"] == scenarios[0].id]
    firsts = [t["room_order"][0] for t in sorted(t0, key=lambda t: t["repeat"])[:4]]
    assert len(set(firsts)) == 4


def test_shells_rotate_and_hide(scenarios, cfg):
    plan = build_plan(cfg, scenarios)
    sc = scenarios[0]
    best = sc.best_room_id
    shells_of_best = {t["shell_map"][best] for t in plan if t["scenario_id"] == sc.id}
    assert len(shells_of_best) == 4      # the best fact profile wears every hotel identity
    t = next(t for t in plan if t["scenario_id"] == sc.id and t["shell_map"][best] != best)
    v = build_view(sc, t)
    room = next(r for r in v["rooms"] if r["room_id"] == best)
    assert room["hotel_name"] == sc.room(t["shell_map"][best])["hotel_name"]
    assert '"shell_id"' not in listings_json(v)


def test_retest_copies_identical_except_seed(scenarios, cfg):
    plan = {t["trial_id"]: t for t in build_plan(cfg, scenarios)}
    rt = [t for t in plan.values() if t["retest"]]
    assert rt
    for t in rt[:20]:
        orig = plan[t["trial_id"].removesuffix("__retest")]
        assert orig["room_order"] == t["room_order"] and orig["room_levels"] == t["room_levels"]
        assert orig["seed"] != t["seed"]


def test_select_trials_spreads_limit(scenarios, cfg):
    from hotel_lab.runner import select_trials
    plan = build_plan(cfg, scenarios)
    sel = select_trials(plan, limit=20)
    assert len(sel) == 20 and len({(t["block"], t["cell"]) for t in sel}) >= 15
    pc = select_trials(plan, per_cell=2)
    cells = {(t["block"], t["cell"]) for t in plan}
    assert len([t for t in pc if not t.get("retest")]) == 2 * len(cells)


def test_damaged_log_line_is_skipped(tmp_path):
    from hotel_lab.runner import read_jsonl
    p = tmp_path / "x.jsonl"
    p.write_text('{"trial_id": "a"}\n{"trial_id": "b', encoding="utf-8")
    assert [r["trial_id"] for r in read_jsonl(p)] == ["a"]


def test_seeds_fit_int32(scenarios, cfg):
    assert max(t["seed"] for t in build_plan(cfg, scenarios)) < 2**31 - 1000


def test_clogit_recovers_effect():
    import numpy as np
    from hotel_lab.analysis import _clogit_cluster
    rng = np.random.default_rng(0)
    y, X, g, c = [], [], [], []
    for s in range(3000):
        x = np.zeros(4)
        x[rng.integers(0, 4)] = 1
        u = 0.8 * x + rng.gumbel(size=4)
        ch = np.argmax(u)
        for j in range(4):
            y.append(float(j == ch)); X.append([x[j]]); g.append(s); c.append(s % 50)
    b, se, G = _clogit_cluster(np.array(y), np.array(X), np.array(g), np.array(c))
    assert abs(b[0] - 0.8) < 3 * se[0] and G == 50


# ---------------------------------------------------------------- Google Sheet database + Studio
def test_sheet_roundtrip(tmp_path, scenarios):
    from hotel_lab import sheetdb
    p = tmp_path / "db.xlsx"
    sheetdb.build_workbook(p, scenarios)
    back = {s.id: s for s in sheetdb.tabs_to_scenarios(sheetdb.read_xlsx(p))}
    for sc in scenarios:
        b = back[sc.id]
        assert b.best_room_id == sc.best_room_id
        for ra, rb in zip(sc.rooms, b.rooms):
            assert ra["facts"] == rb["facts"] and ra["review_sets"] == rb["review_sets"]
            assert {r["review_id"]: r["text"] for r in ra["review_pool"]} == {r["review_id"]: r["text"] for r in rb["review_pool"]}


def test_identical_rooms_only_names_differ(scenarios, cfg):
    from hotel_lab.conditions import build_view
    t = dict(build_plan(cfg, scenarios)[0], identical_rooms=True)
    sc = next(s for s in scenarios if s.id == t["scenario_id"])
    rooms = build_view(sc, t)["rooms"]
    for k in ("price_per_night", "amenities", "guest_rating", "review_count", "room_name", "location"):
        assert len({str(r[k]) for r in rooms}) == 1, k
    assert len({r["hotel_name"] for r in rooms}) == 4


def test_block_configs_and_verdicts_with_simulated_agent():
    """Every hypothesis block plans; a fake agent with a planted emoji preference is detected,
    and the placebo check passes for a random agent."""
    import shutil
    from hotel_lab import RUNS_DIR, study
    from hotel_lab.runner import run
    st = study.load_study()
    st["settings"] = dict(st["settings"], experiment_name="pytest_study", repeats="2", bootstrap_reps="300")
    for b in ("style_emoji", "placebo"):
        info = study.prepare(b, st)
        run(info["experiment"], ["mock-simulated", "mock"], quiet=True)
    res = study.analyze(st, push=False)
    rows = {(r["hypothesis_id"], r["arm"]): r for r in res["rows"]}
    assert rows[("H3a", "mock-simulated__text__low")]["estimate"] > 0
    assert rows[("V7", "mock__text__low")]["verdict"].split(" ")[0] in ("PASS", "FAIL")
    assert rows[("H1", "(all)")]["verdict"] == "Not run yet"
    for d in RUNS_DIR.glob("pytest_study__*"):
        shutil.rmtree(d)


def test_studio_api_smoke():
    from fastapi.testclient import TestClient
    from hotel_lab.studio.app import create_studio
    c = TestClient(create_studio(site_port=None))
    s = c.get("/api/state").json()
    assert s["hypotheses"] and s["models"] and "H3a" in {h["hypothesis_id"] for h in s["hypotheses"]}
    p = c.post("/api/preview", json={"hypothesis_id": "H3a", "seed": 1}).json()
    assert p["rooms"] and "style=" in p["cell"] and p["prompt"]
    assert c.get("/").status_code == 200


def test_studio_models_are_the_four_and_duplicates_are_refused():
    from fastapi.testclient import TestClient
    from hotel_lab.models import MODELS_DIR
    from hotel_lab.studio.app import create_studio
    c = TestClient(create_studio(site_port=None))
    ms = {m["name"]: m for m in c.get("/api/state").json()["models"] if not m["local"]}
    assert set(ms) == {"ollama-qwen2.5-7b", "ollama-llama3.1-8b", "ollama-gemma3-12b", "no-ai-test"}
    assert ms["no-ai-test"]["test_agent"] and not ms["ollama-qwen2.5-7b"]["test_agent"]
    r = c.post("/api/models", json={"name": "qwen-copy", "preset": "ollama", "model": "qwen2.5:7b"})
    assert r.status_code == 409 and "ollama-qwen2.5-7b" in r.json()["detail"]
    assert not (MODELS_DIR / "local-qwen-copy.yaml").exists()
    # an existing copy (e.g. added before this check existed) is flagged, not hidden.
    # Use a temporary copy of the models folder so a running Studio never sees the test file.
    import shutil
    import tempfile
    import hotel_lab.studio.app as studio_app
    tmp = Path(tempfile.mkdtemp())
    for f in MODELS_DIR.glob("*.yaml"):
        shutil.copy(f, tmp / f.name)
    (tmp / "local-pytest-dup.yaml").write_text('provider: ollama\nbase_url: http://localhost:11434/\nmodel: "Qwen2.5:7b"\n',
                                               encoding="utf-8")
    old = studio_app.MODELS_DIR
    studio_app.MODELS_DIR = tmp
    try:
        ms = {m["name"]: m for m in TestClient(create_studio(site_port=None)).get("/api/state").json()["models"]}
        assert ms["local-pytest-dup"]["duplicate_of"] == "ollama-qwen2.5-7b"
        assert "duplicate_of" not in ms["ollama-qwen2.5-7b"]
    finally:
        studio_app.MODELS_DIR = old
        shutil.rmtree(tmp)


def test_models_tab_rows_and_push(monkeypatch):
    from hotel_lab import sheetdb
    rows = sheetdb.model_rows()
    assert [r["name"] for r in rows if r["kind"] == "real model"] and all(set(r) == set(sheetdb.MODEL_COLUMNS) for r in rows)
    assert not any("key" in str(v).lower() and k != "api_key_env" for r in rows for k, v in r.items())
    sent = {}

    class FakeDB:
        def replace(self, tab, rows, columns):
            sent.update(tab=tab, n=len(rows), cols=columns)
            return len(rows)
    assert sheetdb.push_models(FakeDB())["ok"] and sent["tab"] == "Models" and sent["cols"] == sheetdb.MODEL_COLUMNS

    class OldScript:
        def replace(self, *a):
            raise RuntimeError("the website may only write the Results, Verdicts and Runs tabs")
    r = sheetdb.push_models(OldScript())
    assert not r["ok"] and "Apps Script" in r["error"]
    gs = (sheetdb.Path(sheetdb.__file__).parent / "apps_script" / "Code.gs").read_text(encoding="utf-8")
    assert "'Models'" in gs


def test_models_tab_not_in_data_fingerprint(tmp_path, monkeypatch, scenarios):
    from hotel_lab import sheetdb
    monkeypatch.setattr(sheetdb, "DB_DIR", tmp_path)
    monkeypatch.setattr(sheetdb, "CURRENT", tmp_path / "current.json")
    tabs = sheetdb.scenarios_to_tabs(scenarios)
    a = sheetdb.save_snapshot({**tabs, "Models": []}, "t")["sha"]
    b = sheetdb.save_snapshot({**tabs, "Models": [{"name": "x"}]}, "t")["sha"]
    assert a == b
