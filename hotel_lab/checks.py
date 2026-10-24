"""Manipulation checks on the stimuli (brief, step 7) and coder sheets for the checks a script cannot do.

Writes to runs/_checks/:
  variant_stats.csv   words, sentences, emoji count, Flesch reading ease per review x variant
  variant_summary.csv averages per variant + automatic flags
  balance_means.csv   mean rating of each review set per room (balance sets must be equal)
  coder_sheet.csv     random base-vs-variant pairs for two coders: "same facts?" (at least 20 items)
  summary_check.csv   each summary next to its reviews, for two coders: "positives and negatives in proportion?"
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from statistics import mean

import pandas as pd

from hotel_lab import RUNS_DIR
from hotel_lab.stimuli import BALANCE_SETS, Scenario

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF⭐❤]")


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith("le"):
        n -= 1
    return max(n, 1)


def text_stats(text: str) -> dict:
    words = re.findall(r"[A-Za-z']+", text)
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    syll = sum(count_syllables(w) for w in words)
    nw = max(len(words), 1)
    flesch = 206.835 - 1.015 * (nw / sentences) - 84.6 * (syll / nw)
    return {"words": len(words), "sentences": sentences, "emojis": len(EMOJI_RE.findall(text)),
            "chars": len(text), "flesch": round(flesch, 1)}


def run_checks(scenarios: list[Scenario], n_coder_items: int = 24, seed: int = 7) -> Path:
    out = RUNS_DIR / "_checks"
    out.mkdir(parents=True, exist_ok=True)
    rows, bal, pairs, summ = [], [], [], []
    for sc in scenarios:
        for room in sc.rooms:
            pool = {r["review_id"]: r for r in room["review_pool"]}
            facts = "; ".join(f"{f['attribute']}={f['level']}: {f['statement']}" for f in room["facts"])
            for rid in room["review_sets"].get("base", []):
                rv = pool[rid]
                for var, txt in (rv.get("text") or {}).items():
                    rows.append({"scenario": sc.id, "room": room["room_id"], "review_id": rid, "rating": rv["rating"],
                                 "variant": var, **text_stats(txt)})
                    if var != "base":
                        pairs.append({"scenario": sc.id, "room": room["room_id"], "review_id": rid, "variant": var,
                                      "room_facts": facts, "facts_covered": ", ".join(rv.get("facts_covered", [])),
                                      "base_text": rv["text"]["base"], "variant_text": txt,
                                      "coder1_same_facts (Y/N)": "", "coder2_same_facts (Y/N)": "", "notes": ""})
            for sname, ids in room["review_sets"].items():
                ratings = [int(pool[i]["rating"]) for i in ids if i in pool]
                bal.append({"scenario": sc.id, "room": room["room_id"], "set": sname, "n": len(ratings),
                            "mean_rating": round(mean(ratings), 3) if ratings else None,
                            "ratings": " ".join(map(str, sorted(ratings, reverse=True)))})
                if sname in (room.get("summaries") or {}):
                    revs = "\n".join(f"[{pool[i]['rating']}*] {pool[i]['text']['base']}" for i in ids if i in pool)
                    summ.append({"scenario": sc.id, "room": room["room_id"], "set": sname, "room_facts": facts,
                                 "summary": room["summaries"][sname], "summary_words": text_stats(room["summaries"][sname])["words"],
                                 "reviews": revs, "coder1_in_proportion (Y/N)": "", "coder2_in_proportion (Y/N)": "",
                                 "dropped_negatives? (Y/N)": "", "notes": ""})

    vs = pd.DataFrame(rows)
    vs.to_csv(out / "variant_stats.csv", index=False)
    if not vs.empty:
        base = vs[vs["variant"] == "base"].set_index(["scenario", "room", "review_id"])
        summary = vs.groupby("variant")[["words", "sentences", "emojis", "flesch"]].mean().round(2)
        flags = []
        for var, d in vs[vs["variant"] != "base"].groupby("variant"):
            d = d.set_index(["scenario", "room", "review_id"])
            ratio = (d["words"] / base.loc[d.index, "words"]).round(2)
            if "emoji" in var and (d["emojis"] == 0).any():
                flags.append(f"{var}: {(d['emojis'] == 0).sum()} reviews have no emoji")
            if "emoji" not in var and (d["emojis"] > 0).any():
                flags.append(f"{var}: {(d['emojis'] > 0).sum()} reviews contain emoji but should not")
            if "long" in var and (ratio < 1.5).any():
                flags.append(f"{var}: {(ratio < 1.5).sum()} reviews are less than 1.5x the base length")
            if "long" not in var and ((ratio > 1.5) | (ratio < 0.6)).any():
                flags.append(f"{var}: {((ratio > 1.5) | (ratio < 0.6)).sum()} reviews change length a lot (should not)")
        summary["flags"] = ""
        summary.loc[summary.index[0], "flags"] = " | ".join(flags) if flags else "no automatic flags"
        summary.to_csv(out / "variant_summary.csv")
    pd.DataFrame(bal).to_csv(out / "balance_means.csv", index=False)
    random.Random(seed).shuffle(pairs)
    pd.DataFrame(pairs[:n_coder_items]).to_csv(out / "coder_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(out / "summary_check.csv", index=False, encoding="utf-8-sig")
    return out
