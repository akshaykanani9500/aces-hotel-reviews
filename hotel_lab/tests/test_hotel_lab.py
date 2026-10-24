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
