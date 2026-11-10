"""Google Sheet as the study database.

The Google Sheet holds everything you edit (scenarios, rooms, hidden facts, reviews and their variants,
review sets, summaries, hypothesis presets, settings) and receives results (Results, Verdicts, Runs).

How data flows
  Google Sheet --(Apps Script web app, or a downloaded .xlsx)--> sync --> runs/_db/snapshot_<sha>.json
  The experiment always runs from a local SNAPSHOT, never from the live sheet. That makes every run
  reproducible: the snapshot's fingerprint (sha) is stored with every trial. Edit the sheet, press
  "Sync", and the next run uses the new snapshot.

Tabs (one row per thing, values are plain text):
  README, Settings, Hypotheses, Scenarios, Rooms, Facts, Reviews, ReviewSets, Summaries, Models,
  Results, Verdicts, Runs
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hotel_lab import RUNS_DIR, REPO_ROOT

DB_DIR = RUNS_DIR / "_db"
CURRENT = DB_DIR / "current_snapshot.json"
LOCAL_SETTINGS = REPO_ROOT / ".hotel_lab_local.json"     # git-ignored: sheet URL + token live here

TEMPLATE_FIELDS = ("hotel_stars", "room_name", "location", "size_sqm", "bed", "max_guests", "view",
                   "price_per_night", "taxes_and_fees", "amenities", "cancellation", "breakfast",
                   "check_in_out", "guest_rating", "review_count")
INT_FIELDS = {"hotel_stars", "size_sqm", "max_guests", "price_per_night", "taxes_and_fees", "review_count",
              "nights", "rating", "helpful_votes"}
FLOAT_FIELDS = {"guest_rating"}

# ----------------------------------------------------------------------------- hypothesis presets
# One row per hypothesis. `block` = the experiment that produces its data (several hypotheses can share
# a block, e.g. H4 and H4b). Columns factor/levels/assign/control describe the manipulation;
# factor2... a second factor (interaction). Everything not listed stays at its control/default value.
HYPOTHESIS_COLUMNS = [
    "hypothesis_id", "enabled", "family", "block", "maps_to", "question", "H0", "Ha", "predicted",
    "estimand", "contrast_a", "contrast_b", "factor", "levels", "assign", "control",
    "factor2", "levels2", "assign2", "control2", "retest", "sesoi", "pass_threshold", "benchmark", "notes",
]

HYPOTHESES: list[dict[str, Any]] = [
    dict(hypothesis_id="HB", family="gate", block="baseline", maps_to="Brief analysis step 2",
         question="With plain neutral reviews, does the agent pick the fact-best room more often than chance, and does it avoid always picking the same room?",
         H0="Accuracy in the control condition = 25% (chance with 4 rooms)",
         Ha="Accuracy in the control condition differs from 25%", predicted="+",
         estimand="accuracy_vs_chance", factor="order", levels="random", assign="whole_set", control="random",
         sesoi="", pass_threshold="", benchmark="Allouah et al. (2025) rationality checks",
         notes="Gate, not a hypothesis. If it fails for a model, that model's toward/away analyses are descriptive only."),
    dict(hypothesis_id="V1", family="check", block="facts_only", maps_to="Validity: ground truth",
         question="Facts-only calibration: shown the 3 facts as plain labels (no reviews), does the agent pick the fact-best room?",
         H0="", Ha="", predicted="", estimand="threshold_best", factor="format", levels="facts_only",
         assign="whole_set", control="facts_only", pass_threshold="0.80",
         notes="Pass = fact-best room chosen at least 80% of the time. If not, the 'best room' definition is wrong for this agent."),
    dict(hypothesis_id="V5", family="check", block="dominance", maps_to="Validity: attention",
         question="Dominance check: one room is Good on every fact and the others Poor on every fact. Does the agent pick it?",
         H0="", Ha="", predicted="", estimand="threshold_focal", factor="format", levels="facts_only",
         assign="whole_set", control="facts_only", factor2="facts", levels2="all_good", assign2="focal",
         control2="all_poor", pass_threshold="0.95", notes="Pass = dominant room chosen at least 95% of the time."),
    dict(hypothesis_id="V7", family="check", block="placebo", maps_to="Validity: pipeline",
         question="Placebo (A/A): two identical conditions with different labels. Does the pipeline produce a fake effect?",
         H0="No difference between the two identical conditions", Ha="A difference (would mean a pipeline problem)",
         predicted="0", estimand="focal_diff", contrast_a="B", contrast_b="A", factor="placebo", levels="A, B",
         assign="focal", control="A", sesoi="5", pass_threshold="not significant",
         notes="Pass = not significant. The label is never shown to the agent."),
    dict(hypothesis_id="H1", family="confirmatory", block="format", maps_to="Central Q1 / brief H1",
         question="Does a platform-written summary lead to the same choice as the full reviews?",
         H0="Agreement(full vs summary) >= agreement of rerunning the identical page",
         Ha="Agreement(full vs summary) < rerun agreement", predicted="-",
         estimand="agreement_gap", factor="format", levels="full, summary", assign="whole_set", control="full",
         retest="TRUE", sesoi="10", benchmark="None for summaries (stated); Herr, Kardes & Kim (1991) vividness",
         notes="One-sided by construction. Gap = A_FS - mean(A_FF, A_SS) on matched trials."),
    dict(hypothesis_id="H2a", family="confirmatory", block="order", maps_to="Central Q3 / brief H2 (first clause)",
         question="Do platform sort rules (newest, most helpful, most extreme first) change choices compared with random order?",
         H0="No sort rule changes the fact-best room's share (all 3 differences = 0)",
         Ha="At least one sort rule changes it", predicted="none", estimand="best_omnibus",
         factor="order", levels="random, newest, most_helpful, extreme_first", assign="whole_set", control="random",
         sesoi="5", benchmark="Kumar et al. (2026): order sensitivity differs by model"),
    dict(hypothesis_id="H2b", family="confirmatory", block="order_primacy", maps_to="Brief H2 (primacy clause)",
         question="Review ordering: is a room chosen more often when its reviews run best-first than when they run worst-first?",
         H0="P(focal | best first) = P(focal | worst first)", Ha="They differ", predicted="+",
         estimand="focal_diff", contrast_a="best_first", contrast_b="worst_first", factor="order",
         levels="random, best_first, worst_first", assign="focal", control="random", sesoi="5",
         benchmark="Hogarth & Einhorn (1992); Herr, Kardes & Kim (1991)",
         notes="Best-first vs worst-first reverses the whole list, so it captures primacy AND recency. H2c isolates primacy."),
    dict(hypothesis_id="H2c", family="secondary", block="order_primacy", maps_to="Brief H2 (primacy, one-sided contrast)",
         question="Primacy: does putting a room's best review first help it compared with a random order?",
         H0="P(focal | best first) = P(focal | random order)", Ha="They differ", predicted="+",
         estimand="focal_diff", contrast_a="best_first", contrast_b="random", factor="order",
         levels="random, best_first, worst_first", assign="focal", control="random", sesoi="5",
         benchmark="Hogarth & Einhorn (1992)", notes="Shares data with H2b (same block)."),
    dict(hypothesis_id="H3a", family="confirmatory", block="style_emoji", maps_to="Central Q2 / brief H3",
         question="With identical facts and stars, do emojis change how often a room is chosen?",
         H0="P(focal | emoji) = P(focal | no emoji)", Ha="They differ", predicted="none",
         estimand="focal_diff", contrast_a="emoji", contrast_b="base", factor="style", levels="base, emoji",
         assign="focal", control="base", sesoi="5", benchmark="None in the reading list (stated)",
         notes="A null is informative: 'No meaningful effect' needs the 90% CI inside +/-SESOI."),
    dict(hypothesis_id="H3b", family="confirmatory", block="style_length", maps_to="Central Q2 / brief H3",
         question="With identical facts, does a longer review (no new facts) change how often a room is chosen?",
         H0="P(focal | long) = P(focal | short)", Ha="They differ", predicted="+",
         estimand="focal_diff", contrast_a="long", contrast_b="base", factor="style", levels="base, long",
         assign="focal", control="base", sesoi="5",
         benchmark="Hong et al. (2017); Mudambi & Schuff (2010) - helpfulness, not choice"),
    dict(hypothesis_id="H3c", family="confirmatory", block="style_tone", maps_to="Central Q2 / brief H3",
         question="With identical facts, does a casual instead of formal tone change how often a room is chosen?",
         H0="P(focal | casual) = P(focal | formal)", Ha="They differ", predicted="none",
         estimand="focal_diff", contrast_a="casual", contrast_b="base", factor="style", levels="base, casual",
         assign="focal", control="base", sesoi="5", benchmark="None in the reading list (stated)"),
    dict(hypothesis_id="H4", family="confirmatory", block="balance", maps_to="Central Q4 / brief H4",
         question="At the same average rating, do a few strong complaints cost a room more than evenly mixed reviews?",
         H0="P(focal | few strong complaints) = P(focal | evenly mixed)", Ha="They differ", predicted="-",
         estimand="focal_diff", contrast_a="complaints", contrast_b="mixed", factor="review_set",
         levels="mixed, positive, complaints", assign="focal", control="mixed", sesoi="5",
         benchmark="Baumeister et al. (2001); Chevalier & Mayzlin (2006)"),
    dict(hypothesis_id="H4b", family="secondary", block="balance", maps_to="Balance level 'uniformly positive'",
         question="At the same average rating, do uniformly moderate reviews differ from evenly mixed ones?",
         H0="P(focal | uniformly positive) = P(focal | evenly mixed)", Ha="They differ", predicted="+",
         estimand="focal_diff", contrast_a="positive", contrast_b="mixed", factor="review_set",
         levels="mixed, positive, complaints", assign="focal", control="mixed", sesoi="5",
         benchmark="Baumeister et al. (2001)", notes="Shares data with H4 (same block)."),
    dict(hypothesis_id="H5", family="secondary", block="format_x_balance", maps_to="Brief interaction block",
         question="Does the penalty for a few strong complaints change when the agent reads a summary instead of full reviews?",
         H0="Complaint penalty (summary) = complaint penalty (full)", Ha="They differ", predicted="+",
         estimand="focal_did", contrast_a="complaints", contrast_b="mixed", factor="format", levels="full, summary",
         assign="whole_set", control="full", factor2="review_set", levels2="mixed, complaints", assign2="focal",
         control2="mixed", sesoi="5", benchmark="Herr, Kardes & Kim (1991)",
         notes="Low power with 6 scenarios: expect 'inconclusive' unless the effect is large."),
]
for h in HYPOTHESES:
    h.setdefault("enabled", "TRUE")
    for c in HYPOTHESIS_COLUMNS:
        h.setdefault(c, "")

DEFAULT_SETTINGS = [
    ("experiment_name", "study_v1", "Change for every new study, so runs never mix"),
    ("global_seed", "20261110", "Seed for all randomisation (positions, review order, names)"),
    ("items_per_scenario", "4", "Item = which room is focal (rotates over the 4 rooms)"),
    ("repeats", "8", "Per item and cell. Trials per cell = scenarios x items x repeats (6 x 4 x 8 = 192)"),
    ("identical_rooms", "TRUE", "All rooms show the same price, amenities, rating and photo; only reviews differ"),
    ("rotate_shells", "TRUE", "Hotel names rotate across rooms so a name never sticks to the best room"),
    ("position_balance", "latin", "latin = each room in each position equally often"),
    ("prompt_ids", "main", "main, or main, para1, para2 for the paraphrase robustness check"),
    ("interface", "text", "text (JSON) or screenshot (vision models)"),
    ("reasoning_depth", "low", "low or high"),
    ("parse_retries", "1", "Re-ask once if the reply is not valid JSON"),
    ("sesoi_pp", "5", "Equivalence margin in pp. At 192 trials/cell the 90% CI is about +/-7 pp, so a null can only be shown with a margin of ~8-10 or ~850 trials/cell. Decide BEFORE pre-registering"),
    ("alpha", "0.05", "Significance level; Benjamini-Hochberg across the confirmatory family"),
    ("bootstrap_reps", "2000", "Cluster bootstrap repetitions"),
    ("default_order", "random", "Review order when not manipulated"),
    ("default_style", "base", ""),
    ("default_review_set", "base", ""),
    ("default_format", "full", ""),
    ("push_results", "TRUE", "Write each trial to the Results tab"),
    ("api_token", "", "Shared secret for the Apps Script link (filled automatically)"),
]


# ----------------------------------------------------------------------------- YAML -> tabs
def _join(v: Any, sep: str = ", ") -> str:
    if isinstance(v, (list, tuple)):
        return sep.join(str(x) for x in v)
    return "" if v is None else str(v)


def scenarios_to_tabs(scenarios) -> dict[str, list[dict[str, Any]]]:
    tabs: dict[str, list[dict[str, Any]]] = {k: [] for k in ("Scenarios", "Rooms", "Facts", "Reviews", "ReviewSets", "Summaries")}
    for sc in scenarios:
        d = sc.data
        tpl = d["rooms"][0]
        pol = tpl.get("policies") or {}
        tabs["Scenarios"].append({
            "scenario_id": d["scenario_id"], "enabled": "TRUE", "title": d.get("title", ""), "query": d["query"],
            "check_in": d.get("check_in", ""), "nights": d.get("nights", 1), "currency": d.get("currency", "INR"),
            "fact_attributes": _join(d["fact_attributes"]), "best_room_id": d["best_room_id"],
            **{k: tpl.get(k, "") for k in ("hotel_stars", "room_name", "location", "size_sqm", "bed", "max_guests",
                                           "view", "price_per_night", "taxes_and_fees")},
            "amenities": _join(tpl.get("amenities", []), "; "),
            "cancellation": pol.get("cancellation", ""), "breakfast": pol.get("breakfast", ""),
            "check_in_out": pol.get("check_in_out", ""),
            "guest_rating": tpl.get("guest_rating", ""), "review_count": tpl.get("review_count", ""),
        })
        for r in d["rooms"]:
            tabs["Rooms"].append({"scenario_id": d["scenario_id"], "room_id": r["room_id"], "hotel_name": r["hotel_name"]})
            for f in r["facts"]:
                tabs["Facts"].append({"scenario_id": d["scenario_id"], "room_id": r["room_id"],
                                      "attribute": f["attribute"], "level": f["level"], "statement": f.get("statement", "")})
            for rv in r["review_pool"]:
                row = {"scenario_id": d["scenario_id"], "room_id": r["room_id"], "review_id": rv["review_id"],
                       **{k: rv.get(k, "") for k in ("author", "country", "traveller_type", "stayed", "posted_date",
                                                     "helpful_votes", "rating", "title")},
                       "facts_covered": _join(rv.get("facts_covered", []))}
                for key, txt in (rv.get("text") or {}).items():
                    row[f"text_{key}"] = txt
                tabs["Reviews"].append(row)
            for sname, ids in r["review_sets"].items():
                tabs["ReviewSets"].append({"scenario_id": d["scenario_id"], "room_id": r["room_id"],
                                           "set_name": sname, "review_ids": _join(ids)})
            for sname, txt in (r.get("summaries") or {}).items():
                tabs["Summaries"].append({"scenario_id": d["scenario_id"], "room_id": r["room_id"],
                                          "set_name": sname, "summary": txt})
    # make sure the standard variant columns exist and come in a fixed order
    return tabs


# ----------------------------------------------------------------------------- tabs -> scenarios
def _val(v: Any, key: str) -> Any:
    if v is None:
        return ""
    s = str(v).strip()
    if key in INT_FIELDS and s != "":
        try:
            return int(float(s.replace(",", "")))
        except ValueError:
            return s
    if key in FLOAT_FIELDS and s != "":
        try:
            return float(s)
        except ValueError:
            return s
    return s


def _truthy(v: Any) -> bool:
    return str(v).strip().upper() in ("TRUE", "YES", "Y", "1")


DATE_FIELDS = {"posted_date", "check_in"}


def _norm_cell(v: Any) -> Any:
    """Excel/Sheets dates and whole numbers -> plain text, so JSON and the pages never see datetimes."""
    import datetime as _dt
    if isinstance(v, _dt.datetime):
        return v.strftime("%Y-%m-%d") if (v.hour, v.minute, v.second) == (0, 0, 0) else v.isoformat(sep=" ")
    if isinstance(v, _dt.date):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return v


def norm_date(s: Any) -> str:
    """Accept 2026-09-18, 18/09/2026, 18-09-2026, 2026-09-18 00:00:00 -> 2026-09-18 (day first, as in India)."""
    import re as _re
    s = str(s).strip()
    m = _re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
    m = _re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", s)
    if m:
        return f"{int(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}"
    return s


def rows_from_grid(grid: list[list[Any]]) -> list[dict[str, Any]]:
    """Header row + data rows -> list of dicts. Blank rows are skipped; dates become YYYY-MM-DD."""
    if not grid:
        return []
    header = [str(h).strip() for h in grid[0]]
    out = []
    for row in grid[1:]:
        if not any(str(c).strip() for c in row):
            continue
        d = {h: _norm_cell(row[i] if i < len(row) else "") for i, h in enumerate(header) if h}
        for k in DATE_FIELDS & d.keys():
            d[k] = norm_date(d[k])
        out.append(d)
    return out


def tabs_to_scenarios(tabs: dict[str, list[dict[str, Any]]]):
    """Rebuild the scenario dicts the engine uses (same shape as the YAML files)."""
    from hotel_lab.stimuli import Scenario
    by_sc: dict[str, dict[str, Any]] = {}
    for s in tabs.get("Scenarios", []):
        if not s.get("scenario_id") or not _truthy(s.get("enabled", "TRUE")):
            continue
        sid = str(s["scenario_id"]).strip()
        tpl = {k: _val(s.get(k, ""), k) for k in TEMPLATE_FIELDS}
        amen = [a.strip() for a in str(s.get("amenities", "")).split(";") if a.strip()]
        policies = {k: str(s.get(k, "")).strip() for k in ("cancellation", "breakfast", "check_in_out") if str(s.get(k, "")).strip()}
        by_sc[sid] = {
            "scenario_id": sid, "title": str(s.get("title", "")), "query": str(s.get("query", "")),
            "check_in": str(s.get("check_in", "")), "nights": _val(s.get("nights", 1), "nights") or 1,
            "currency": str(s.get("currency", "INR")) or "INR",
            "fact_attributes": [a.strip() for a in str(s.get("fact_attributes", "")).split(",") if a.strip()],
            "best_room_id": str(s.get("best_room_id", "")).strip(),
            "_template": {**{k: v for k, v in tpl.items() if k not in ("amenities", "cancellation", "breakfast", "check_in_out")},
                          "amenities": amen, "policies": policies},
            "rooms": {},
        }

    def room(sid: str, rid: str) -> Optional[dict[str, Any]]:
        sc = by_sc.get(sid)
        if sc is None:
            return None
        if rid not in sc["rooms"]:
            sc["rooms"][rid] = {"room_id": rid, **sc["_template"], "hotel_name": rid, "facts": [],
                                "review_pool": [], "review_sets": {}, "summaries": {}}
        return sc["rooms"][rid]

    for r in tabs.get("Rooms", []):
        rm = room(str(r.get("scenario_id", "")).strip(), str(r.get("room_id", "")).strip())
        if rm is not None:
            rm["hotel_name"] = str(r.get("hotel_name", "")).strip() or rm["room_id"]
    for f in tabs.get("Facts", []):
        rm = room(str(f.get("scenario_id", "")).strip(), str(f.get("room_id", "")).strip())
        if rm is not None:
            rm["facts"].append({"attribute": str(f.get("attribute", "")).strip(),
                                "level": str(f.get("level", "")).strip().lower(),
                                "statement": str(f.get("statement", "")).strip()})
    for rv in tabs.get("Reviews", []):
        rm = room(str(rv.get("scenario_id", "")).strip(), str(rv.get("room_id", "")).strip())
        if rm is None or not str(rv.get("review_id", "")).strip():
            continue
        texts = {k[5:]: str(v) for k, v in rv.items() if k.startswith("text_") and str(v).strip()}
        rm["review_pool"].append({
            "review_id": str(rv["review_id"]).strip(),
            **{k: _val(rv.get(k, ""), k) for k in ("author", "country", "traveller_type", "stayed", "posted_date",
                                                  "helpful_votes", "rating", "title")},
            "facts_covered": [a.strip() for a in str(rv.get("facts_covered", "")).split(",") if a.strip()],
            "text": texts,
        })
    for s in tabs.get("ReviewSets", []):
        rm = room(str(s.get("scenario_id", "")).strip(), str(s.get("room_id", "")).strip())
        if rm is not None and str(s.get("set_name", "")).strip():
            rm["review_sets"][str(s["set_name"]).strip()] = [i.strip() for i in str(s.get("review_ids", "")).split(",") if i.strip()]
    for s in tabs.get("Summaries", []):
        rm = room(str(s.get("scenario_id", "")).strip(), str(s.get("room_id", "")).strip())
        if rm is not None and str(s.get("set_name", "")).strip():
            rm["summaries"][str(s["set_name"]).strip()] = str(s.get("summary", "")).strip()

    out = []
    for sid, sc in by_sc.items():
        sc = dict(sc)
        sc.pop("_template")
        sc["rooms"] = [sc["rooms"][k] for k in sorted(sc["rooms"])]
        out.append(Scenario(data=sc, path=CURRENT))
    return out


def tabs_to_hypotheses(tabs) -> list[dict[str, Any]]:
    out = []
    for h in tabs.get("Hypotheses", []):
        if not str(h.get("hypothesis_id", "")).strip():
            continue
        h = {k: str(v).strip() for k, v in h.items()}
        h["enabled"] = _truthy(h.get("enabled", "TRUE"))
        out.append(h)
    return out


def tabs_to_settings(tabs) -> dict[str, str]:
    s = {k: v for k, v, _ in DEFAULT_SETTINGS}
    for row in tabs.get("Settings", []):
        k = str(row.get("key", "")).strip()
        if k:
            s[k] = str(row.get("value", "")).strip()
    return s


# ----------------------------------------------------------------------------- workbook (.xlsx)
README_ROWS = [
    ["Hotel Lab study database", ""],
    ["What this is", "The single source of truth for the experiment. The local website reads it, runs the AI agents, and writes results back here."],
    ["Edit freely", "Scenarios, Rooms, Facts, Reviews, ReviewSets, Summaries, Hypotheses, Settings, Models"],
    ["Written by the website", "Results (one row per trial), Verdicts (one row per hypothesis x model), Runs (one row per run). Do not edit these."],
    ["Settings", "Key/value pairs: sample size, seeds, equivalence margin, defaults for factors that are not being tested."],
    ["Hypotheses", "One row per hypothesis. The website's dropdown is built from this tab. 'block' says which experiment produces the data."],
    ["Scenarios", "One row per trip scenario (6). The room template (price, amenities, rating...) is here: every room on the page is identical."],
    ["Rooms", "Hotel name per room. Names rotate across rooms during the experiment."],
    ["Facts", "HIDDEN ground truth: 3 attributes per room, level good / average / poor. Never shown to the agent."],
    ["Reviews", "Every review a room can show. text_base is the plain version; text_emoji, text_long, text_casual are style variants (same facts, same stars)."],
    ["ReviewSets", "Which 8 reviews make up each set: base (= mixed), positive, complaints. Mean rating must be equal across positive/mixed/complaints."],
    ["Summaries", "Platform-written summary for each review set (format hypothesis H1)."],
    ["Connect", "Extensions -> Apps Script -> paste hotel_lab/apps_script/Code.gs from the repo -> Deploy -> New deployment -> Web app (Execute as: Me, Who has access: Anyone) -> copy the URL into the website."],
    ["After editing", "Press 'Sync from Sheet' in the website. Runs always use a frozen local snapshot, so a run is never affected by edits made while it is running."],
]


def build_workbook(path: Path, scenarios, models: list[dict[str, Any]] | None = None, token: str | None = None) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    tabs = scenarios_to_tabs(scenarios)
    token = token or secrets.token_urlsafe(16)
    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for r in README_ROWS:
        ws.append(r)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 140
    ws["A1"].font = Font(bold=True, size=14)
    for row in ws.iter_rows(min_row=2):
        row[0].font = Font(bold=True)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    head_fill = PatternFill("solid", fgColor="0B5C8F")

    def add(name: str, rows: list[dict[str, Any]], columns: list[str] | None = None, widths: dict[str, int] | None = None):
        sh = wb.create_sheet(name)
        cols = columns or list(dict.fromkeys(k for r in rows for k in r))
        sh.append(cols)
        for r in rows:
            sh.append(["" if r.get(c) is None else str(r.get(c, "")) for c in cols])
        for i, c in enumerate(cols, start=1):
            cell = sh.cell(row=1, column=i)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = head_fill
            letter = cell.column_letter
            sh.column_dimensions[letter].width = (widths or {}).get(c, 60 if c.startswith(("text_", "summary", "question", "H0", "Ha", "notes", "query")) else 16)
        sh.freeze_panes = "A2"
        # every value is stored as text, so Google Sheets keeps "2026-09-18" and "0.10" exactly
        for row in sh.iter_rows(min_row=2):
            for cell in row:
                cell.number_format = "@"
        return sh, cols

    settings = [{"key": k, "value": (token if k == "api_token" else v), "meaning": m} for k, v, m in DEFAULT_SETTINGS]
    add("Settings", settings, ["key", "value", "meaning"], {"meaning": 90, "value": 24, "key": 22})
    add("Hypotheses", HYPOTHESES, HYPOTHESIS_COLUMNS)
    sh, cols = add("Scenarios", tabs["Scenarios"])
    add("Rooms", tabs["Rooms"])
    sh, cols = add("Facts", tabs["Facts"])
    dv = DataValidation(type="list", formula1='"good,average,poor"', allow_blank=False)
    sh.add_data_validation(dv)
    dv.add(f"D2:D{max(len(tabs['Facts']) + 200, 300)}")
    std = ["scenario_id", "room_id", "review_id", "author", "country", "traveller_type", "stayed", "posted_date",
           "helpful_votes", "rating", "title", "facts_covered", "text_base", "text_emoji", "text_long", "text_casual"]
    extra = [c for c in dict.fromkeys(k for r in tabs["Reviews"] for k in r) if c not in std]
    add("Reviews", tabs["Reviews"], std + extra)
    add("ReviewSets", tabs["ReviewSets"], ["scenario_id", "room_id", "set_name", "review_ids"], {"review_ids": 80})
    add("Summaries", tabs["Summaries"], ["scenario_id", "room_id", "set_name", "summary"])
    model_cols = ["name", "provider", "base_url", "model", "api_key_env", "family", "size_b", "quantisation",
                  "serving", "temperature", "max_tokens", "num_ctx", "supports_vision", "notes"]
    add("Models", models or [], model_cols)
    add("Results", [], RESULT_COLUMNS)
    add("Verdicts", [], VERDICT_COLUMNS)
    add("Runs", [], RUN_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


RESULT_COLUMNS = ["timestamp", "run_id", "experiment", "hypothesis_block", "trial_id", "cell", "scenario_id", "item",
                  "repeat", "prompt_id", "retest", "model_name", "model", "interface", "reasoning_depth",
                  "temperature", "focal_room_id", "best_room_id", "room_order", "chosen_option", "chosen_room_id",
                  "chose_best", "chose_focal", "parse_ok", "attempts", "error", "reason", "input_tokens",
                  "output_tokens", "latency_s", "seed", "snapshot_sha"]
VERDICT_COLUMNS = ["timestamp", "experiment", "hypothesis_id", "family", "arm", "estimand", "n_trials", "clusters",
                   "min_cell_n", "failure_rate",
                   "estimate", "unit", "ci95_low", "ci95_high", "ci90_low", "ci90_high", "p", "q_bh", "sesoi",
                   "significant", "equivalent", "verdict", "decision_H0", "direction", "predicted",
                   "heterogeneity_p", "plain_english", "notes", "max_room_share", "max_position_share"]
RUN_COLUMNS = ["timestamp", "run_id", "experiment", "block", "models", "trials_planned", "trials_done",
               "failures", "snapshot_sha", "git_commit", "status", "notes"]


def read_xlsx(path: Path) -> dict[str, list[dict[str, Any]]]:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    out = {}
    for ws in wb.worksheets:
        grid = [["" if c is None else c for c in row] for row in ws.iter_rows(values_only=True)]
        out[ws.title] = rows_from_grid(grid) if ws.title != "README" else []
    return out


# ----------------------------------------------------------------------------- Apps Script backend
class AppsScriptDB:
    """Talks to the Apps Script web app pasted into the Sheet (hotel_lab/apps_script/Code.gs)."""

    def __init__(self, url: str, token: str, timeout: float = 60):
        self.url, self.token, self.timeout = url.strip(), token.strip(), timeout

    def _client(self):
        import httpx
        return httpx.Client(timeout=self.timeout, follow_redirects=True)

    def read_all(self) -> dict[str, list[dict[str, Any]]]:
        with self._client() as c:
            r = c.get(self.url, params={"action": "read", "token": self.token})
        data = self._json(r)
        return {name: rows_from_grid(grid) for name, grid in data["tabs"].items() if name != "README"}

    def append(self, tab: str, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        with self._client() as c:
            r = c.post(self.url, content=json.dumps({"action": "append", "token": self.token, "tab": tab,
                                                     "rows": rows}, default=str),
                       headers={"Content-Type": "text/plain;charset=utf-8"})
        return int(self._json(r).get("appended", 0))

    def replace(self, tab: str, rows: list[dict[str, Any]], columns: list[str]) -> int:
        with self._client() as c:
            r = c.post(self.url, content=json.dumps({"action": "replace", "token": self.token, "tab": tab,
                                                     "columns": columns, "rows": rows}, default=str),
                       headers={"Content-Type": "text/plain;charset=utf-8"})
        return int(self._json(r).get("written", 0))

    @staticmethod
    def _json(r) -> dict[str, Any]:
        if r.status_code != 200:
            raise RuntimeError(f"Sheet link returned HTTP {r.status_code}. Check the URL and that the "
                               f"deployment has access 'Anyone'.")
        try:
            data = r.json()
        except Exception:
            raise RuntimeError("The Sheet link did not return JSON. Usually the deployment is not set to "
                               "'Anyone', or you copied the editor URL instead of the /exec web app URL.")
        if not data.get("ok"):
            raise RuntimeError(f"Sheet error: {data.get('error')}")
        return data


# ----------------------------------------------------------------------------- local settings + snapshot
def load_local_settings() -> dict[str, Any]:
    try:
        return json.loads(LOCAL_SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_local_settings(**kw: Any) -> dict[str, Any]:
    s = load_local_settings()
    s.update({k: v for k, v in kw.items() if v is not None})
    LOCAL_SETTINGS.write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def get_db() -> Optional[AppsScriptDB]:
    s = load_local_settings()
    url = os.getenv("HOTEL_LAB_SHEET_URL") or s.get("sheet_url")
    token = os.getenv("HOTEL_LAB_SHEET_TOKEN") or s.get("sheet_token")
    return AppsScriptDB(url, token or "") if url else None


def save_snapshot(tabs: dict[str, list[dict[str, Any]]], source: str) -> dict[str, Any]:
    """Validate and freeze the sheet contents locally. Returns snapshot info (raises if invalid)."""
    from hotel_lab.stimuli import validate_all
    scenarios = tabs_to_scenarios(tabs)
    if not scenarios:
        raise ValueError("The Scenarios tab has no enabled scenarios.")
    styles = sorted({k for sc in scenarios for r in sc.rooms for rv in r["review_pool"] for k in rv["text"]})
    report = validate_all(scenarios, style_levels=[s for s in ("emoji", "long", "casual") if s in styles],
                          summary_sets=["base", "positive", "mixed", "complaints"])
    content = {k: v for k, v in tabs.items() if k not in ("Results", "Verdicts", "Runs")}
    sha = hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()[:16]
    DB_DIR.mkdir(parents=True, exist_ok=True)
    snap = {"sha": sha, "source": source, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tabs": content, "validation": {"errors": report.errors, "warnings": report.warnings}}
    (DB_DIR / f"snapshot_{sha}.json").write_text(json.dumps(snap, ensure_ascii=False, default=str), encoding="utf-8")
    if report.ok:
        CURRENT.write_text(json.dumps(snap, ensure_ascii=False, default=str), encoding="utf-8")
    return {"sha": sha, "ok": report.ok, "source_type": source, "errors": report.errors, "warnings": report.warnings,
            "scenarios": [s.id for s in scenarios], "fetched_at": snap["fetched_at"], "source": source}


def current_snapshot() -> Optional[dict[str, Any]]:
    if not CURRENT.exists():
        return None
    return json.loads(CURRENT.read_text(encoding="utf-8"))


def sync_from_sheet() -> dict[str, Any]:
    db = get_db()
    if db is None:
        raise RuntimeError("No Sheet link saved yet. Paste the Apps Script web-app URL and token first.")
    return save_snapshot(db.read_all(), source="google_sheet")


def sync_from_xlsx(path: Path) -> dict[str, Any]:
    return save_snapshot(read_xlsx(Path(path)), source=f"xlsx:{Path(path).name}")
