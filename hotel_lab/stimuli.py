"""Load and validate scenario files (hotel rooms, hidden fact sheets, reviews, summaries).

See hotel_lab/stimuli/SCHEMA.md for the file format.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from hotel_lab import STIMULI_DIR

FACT_SCORE = {"good": 2, "average": 1, "poor": 0}
BALANCE_SETS = ("positive", "mixed", "complaints")
REQUIRED_ROOM_FIELDS = (
    "room_id", "hotel_name", "room_name", "price_per_night", "guest_rating",
    "review_count", "facts", "review_pool", "review_sets",
)


@dataclass
class Scenario:
    data: dict[str, Any]
    path: Path

    @property
    def id(self) -> str:
        return self.data["scenario_id"]

    @property
    def rooms(self) -> list[dict[str, Any]]:
        return self.data["rooms"]

    @property
    def best_room_id(self) -> str:
        return self.data["best_room_id"]

    def room(self, room_id: str) -> dict[str, Any]:
        for r in self.rooms:
            if r["room_id"] == room_id:
                return r
        raise KeyError(f"{self.id}: no room {room_id}")

    def review(self, room: dict[str, Any], review_id: str) -> dict[str, Any]:
        for rv in room["review_pool"]:
            if rv["review_id"] == review_id:
                return rv
        raise KeyError(f"{self.id}/{room['room_id']}: no review {review_id}")

    def fact_score(self, room: dict[str, Any]) -> int:
        return sum(FACT_SCORE[f["level"]] for f in room["facts"])


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "ValidationReport") -> None:
        self.errors += other.errors
        self.warnings += other.warnings


def load_scenarios(stimuli_dir: Path | str = STIMULI_DIR, only: list[str] | None = None) -> list[Scenario]:
    """Load every *.yaml in the stimuli folder (sorted by name)."""
    out = []
    for p in sorted(Path(stimuli_dir).glob("*.yaml")):
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        sc = Scenario(data=data, path=p)
        if only and sc.id not in only:
            continue
        out.append(sc)
    if not out:
        raise FileNotFoundError(f"No scenario .yaml files found in {stimuli_dir}")
    return out


def validate_scenario(sc: Scenario, style_levels: list[str] | None = None,
                      summary_sets: list[str] | None = None) -> ValidationReport:
    """Mechanical checks. Human coders still need to check that variants carry the same facts."""
    rep = ValidationReport()
    d = sc.data
    tag = sc.path.name
    for key in ("scenario_id", "query", "rooms", "best_room_id", "fact_attributes"):
        if key not in d:
            rep.errors.append(f"{tag}: missing top-level key '{key}'")
    if rep.errors:
        return rep

    attrs = d["fact_attributes"]
    rooms = d["rooms"]
    ids = [r.get("room_id") for r in rooms]
    if len(rooms) != 4:
        rep.warnings.append(f"{tag}: {len(rooms)} rooms (design assumes 4)")
    if len(set(ids)) != len(ids):
        rep.errors.append(f"{tag}: duplicate room_id")
    if d["best_room_id"] not in ids:
        rep.errors.append(f"{tag}: best_room_id '{d['best_room_id']}' is not a room")

    scores = {}
    for r in rooms:
        rt = f"{tag}/{r.get('room_id')}"
        for k in REQUIRED_ROOM_FIELDS:
            if k not in r:
                rep.errors.append(f"{rt}: missing '{k}'")
        if any(k not in r for k in REQUIRED_ROOM_FIELDS):
            continue
        fact_attrs = [f["attribute"] for f in r["facts"]]
        if sorted(fact_attrs) != sorted(attrs):
            rep.errors.append(f"{rt}: facts {fact_attrs} do not match fact_attributes {attrs}")
        for f in r["facts"]:
            if f.get("level") not in FACT_SCORE:
                rep.errors.append(f"{rt}: fact level must be good/average/poor, got {f.get('level')}")
        if all(f.get("level") in FACT_SCORE for f in r["facts"]):
            scores[r["room_id"]] = sc.fact_score(r)

        pool = {rv["review_id"]: rv for rv in r["review_pool"]}
        if len(pool) != len(r["review_pool"]):
            rep.errors.append(f"{rt}: duplicate review_id in pool")
        for rv in r["review_pool"]:
            if not (1 <= int(rv.get("rating", 0)) <= 5):
                rep.errors.append(f"{rt}/{rv['review_id']}: rating must be 1-5")
            if "base" not in (rv.get("text") or {}):
                rep.errors.append(f"{rt}/{rv['review_id']}: missing text.base")
            for a in rv.get("facts_covered", []):
                if a not in attrs:
                    rep.errors.append(f"{rt}/{rv['review_id']}: facts_covered '{a}' not in fact_attributes")
            for k in ("posted_date", "helpful_votes"):
                if k not in rv:
                    rep.errors.append(f"{rt}/{rv['review_id']}: missing '{k}' (needed for order conditions)")

        sets = r["review_sets"]
        if "base" not in sets:
            rep.errors.append(f"{rt}: review_sets needs a 'base' set")
        for sname, sids in sets.items():
            missing = [i for i in sids if i not in pool]
            if missing:
                rep.errors.append(f"{rt}: set '{sname}' has unknown ids {missing}")
            if len(sids) != 8:
                rep.warnings.append(f"{rt}: set '{sname}' has {len(sids)} reviews (design assumes 8)")
            if len(set(sids)) != len(sids):
                rep.errors.append(f"{rt}: set '{sname}' repeats a review")

        present_balance = [s for s in BALANCE_SETS if s in sets]
        if present_balance:
            if len(present_balance) != 3:
                rep.errors.append(f"{rt}: has {present_balance}; balance block needs all of {BALANCE_SETS}")
            else:
                means = {s: mean(int(pool[i]["rating"]) for i in sets[s] if i in pool) for s in BALANCE_SETS}
                if max(means.values()) - min(means.values()) > 0.01:
                    rep.errors.append(f"{rt}: balance sets must have equal mean rating, got {means}")

        for lvl in style_levels or []:
            if lvl == "base":
                continue
            for i in sets.get("base", []):
                if i in pool and lvl not in (pool[i].get("text") or {}):
                    rep.errors.append(f"{rt}/{i}: missing style variant '{lvl}' (used in experiment config)")

        for s in summary_sets or []:
            if s in sets and s not in (r.get("summaries") or {}):
                rep.errors.append(f"{rt}: missing summary for set '{s}'")

    if scores and d["best_room_id"] in scores:
        best = scores[d["best_room_id"]]
        others = [v for k, v in scores.items() if k != d["best_room_id"]]
        if others and best <= max(others):
            rep.errors.append(f"{tag}: best_room_id is not strictly best on the fact sheet (scores {scores})")
        elif others and best - max(others) > 1:
            rep.warnings.append(f"{tag}: best room wins by {best - max(others)} points; brief asks for a moderate margin")

    prices = [r["price_per_night"] for r in rooms if "price_per_night" in r]
    if prices and (max(prices) - min(prices)) / min(prices) > 0.10:
        rep.warnings.append(f"{tag}: prices differ by more than 10% across rooms; price may swamp review effects")
    ratings = {r.get("guest_rating") for r in rooms}
    if len(ratings) > 1:
        rep.warnings.append(f"{tag}: guest_rating differs across rooms {sorted(ratings)}; stars are a strong cue")
    return rep


def validate_all(scenarios: list[Scenario], style_levels=None, summary_sets=None) -> ValidationReport:
    rep = ValidationReport()
    seen = set()
    for sc in scenarios:
        if sc.id in seen:
            rep.errors.append(f"duplicate scenario_id {sc.id}")
        seen.add(sc.id)
        rep.extend(validate_scenario(sc, style_levels, summary_sets))
    return rep
