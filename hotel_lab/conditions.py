"""Apply a trial's factor levels to a scenario and produce exactly what the agent will see.

The same `build_view` output feeds the text (headless) prompt, the website, and the screenshots,
so the three interfaces always show identical content.
"""
from __future__ import annotations

import random
from typing import Any

from hotel_lab.design import stable_seed
from hotel_lab.stimuli import Scenario

# Fields copied from the stimulus file to the page. `facts` are NEVER copied (hidden ground truth).
# All of these belong to the room's "shell" (visible identity). With rotate_shells, a fact profile
# (facts + reviews + summaries) can wear another room's shell.
VISIBLE_ROOM_FIELDS = (
    "hotel_name", "hotel_stars", "room_name", "location", "size_sqm", "bed", "max_guests", "view",
    "price_per_night", "taxes_and_fees", "amenities", "policies", "guest_rating", "review_count",
)
VISIBLE_REVIEW_FIELDS = ("author", "country", "traveller_type", "stayed", "posted_date", "helpful_votes", "rating", "title")


def order_reviews(reviews: list[dict[str, Any]], level: str, rng: random.Random) -> list[dict[str, Any]]:
    """Return the same reviews in a new order. Content never changes here."""
    rv = reviews[:]
    rng.shuffle(rv)  # random tie-breaking for every rule below (sorts are stable)
    if level == "random":
        return rv
    if level == "newest":
        return sorted(rv, key=lambda r: r["posted_date"], reverse=True)
    if level == "oldest":
        return sorted(rv, key=lambda r: r["posted_date"])
    if level == "most_helpful":
        return sorted(rv, key=lambda r: int(r["helpful_votes"]), reverse=True)
    if level == "extreme_first":  # distance from the scale midpoint (3 stars)
        return sorted(rv, key=lambda r: abs(int(r["rating"]) - 3), reverse=True)
    if level == "best_first":
        return sorted(rv, key=lambda r: int(r["rating"]), reverse=True)
    if level == "worst_first":
        return sorted(rv, key=lambda r: int(r["rating"]))
    raise ValueError(f"unknown order level '{level}'")


def review_text(review: dict[str, Any], style: str) -> str:
    texts = review.get("text") or {}
    if style in texts:
        return texts[style]
    raise KeyError(
        f"review {review['review_id']} has no '{style}' text variant. "
        f"Add it to the stimulus file or remove the level from experiment.yaml."
    )


def room_view(sc: Scenario, room: dict[str, Any], levels: dict[str, str], seed: int,
              show_category_scores: bool = False, shell: dict[str, Any] | None = None) -> dict[str, Any]:
    """`room` supplies facts/reviews/summaries; `shell` (default: the same room) supplies what the page shows."""
    shell = shell or room
    rng = random.Random(stable_seed(seed, room["room_id"], "order"))
    set_name = levels["review_set"]
    if set_name not in room["review_sets"]:
        raise KeyError(f"{sc.id}/{room['room_id']}: no review set '{set_name}'")
    pool = [sc.review(room, i) for i in room["review_sets"][set_name]]
    ordered = order_reviews(pool, levels["order"], rng)

    view = {k: shell[k] for k in VISIBLE_ROOM_FIELDS if k in shell}
    view["room_id"] = room["room_id"]
    view["shell_id"] = shell["room_id"]
    view["total_price"] = shell["price_per_night"] * sc.data.get("nights", 1) + shell.get("taxes_and_fees", 0)
    if show_category_scores and "category_scores" in room:
        view["category_scores"] = room["category_scores"]

    fmt = levels["format"]
    if fmt in ("full", "both"):
        view["reviews"] = [
            {**{k: r[k] for k in VISIBLE_REVIEW_FIELDS if k in r},
             "review_id": r["review_id"],
             "text": review_text(r, levels["style"])}
            for r in ordered
        ]
    if fmt in ("summary", "both"):
        summaries = room.get("summaries") or {}
        if set_name not in summaries:
            raise KeyError(f"{sc.id}/{room['room_id']}: no summary for review set '{set_name}'")
        view["review_summary"] = summaries[set_name]
    view["levels"] = dict(levels)
    return view


def build_view(sc: Scenario, trial: dict[str, Any], show_category_scores: bool = False) -> dict[str, Any]:
    """Everything visible in one trial, rooms listed in the trial's shuffled display order."""
    rooms = []
    seed = trial.get("layout_seed", trial["seed"])  # shared across cells -> same random review order when paired
    shells = trial.get("shell_map") or {}
    for pos, rid in enumerate(trial["room_order"], start=1):
        shell = sc.room(shells[rid]) if rid in shells else None
        v = room_view(sc, sc.room(rid), trial["room_levels"][rid], seed, show_category_scores, shell)
        v["option_number"] = pos
        rooms.append(v)
    return {
        "scenario_id": sc.id,
        "title": sc.data.get("title", sc.id),
        "query": sc.data["query"],
        "check_in": sc.data.get("check_in"),
        "nights": sc.data.get("nights", 1),
        "currency": sc.data.get("currency", "INR"),
        "rooms": rooms,
    }


def preview_trial(sc: Scenario, levels: dict[str, str], seed: int = 0, shuffle: bool = False) -> dict[str, Any]:
    """A one-off trial for looking at pages in the browser (same levels for every room)."""
    ids = [r["room_id"] for r in sc.rooms]
    if shuffle:
        random.Random(seed).shuffle(ids)
    return {
        "trial_id": f"preview__{sc.id}",
        "seed": seed,
        "room_order": ids,
        "room_levels": {rid: dict(levels) for rid in ids},
        "block": "preview",
        "cell": "preview",
        "scenario_id": sc.id,
    }
