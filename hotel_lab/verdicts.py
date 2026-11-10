"""Turn trial results into one plain verdict per hypothesis (and per model).

Everything is on ONE scale: percentage points (pp) of choice share (or agreement, for H1).
Uncertainty comes from a cluster bootstrap. A cluster = one scenario x focal room (6 x 4 = 24 units),
because trials that share the same reviews are not independent.

For every hypothesis we report:
  estimate (pp), 95% CI, 90% CI, p-value, q (Benjamini-Hochberg across the confirmatory family),
  and a verdict from two questions (Lakens' 2 x 2):
     significant?  = p (or q) < alpha              -> can we reject H0?
     equivalent?   = 90% CI inside +/- SESOI      -> is any effect too small to matter?
       significant & not equivalent -> "Effect"                    (reject H0)
       significant & equivalent     -> "Real but negligible"       (reject H0, effect is tiny)
       not sig.    & equivalent     -> "No meaningful effect"      (an informative null)
       not sig.    & not equivalent -> "Inconclusive"              (need more data)
Checks (V1, V5) and the gate (HB) report pass/fail instead.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from hotel_lab import RUNS_DIR


def block_experiment(study: str, block: str) -> str:
    return f"{study}__{block}"


def _sesoi_pp(h: dict[str, Any], default_pp: float) -> float:
    try:
        v = float(h.get("sesoi") or default_pp)
    except ValueError:
        v = default_pp
    return v * 100 if v < 1 else v      # "0.10" means 10 pp


def _weights(G: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap as a B x G matrix of how often each cluster is drawn."""
    return rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(float)


def _ratio(k: np.ndarray, n: np.ndarray, W: np.ndarray) -> np.ndarray:
    num, den = W @ k, W @ n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _cells(df: pd.DataFrame, clusters: list[str], mask: pd.Series, col: str) -> tuple[np.ndarray, np.ndarray]:
    d = df[mask]
    g = d.groupby("item_id")[col]
    k = g.sum().reindex(clusters, fill_value=0).values.astype(float)
    n = g.size().reindex(clusters, fill_value=0).values.astype(float)
    return k, n


# ------------------------------------------------------------------ estimands: (point, boot vector)
def est_focal_diff(df, clusters, W, h):
    col = f"cell_{h['factor']}"
    ka, na = _cells(df, clusters, df[col] == h["contrast_a"], "chose_focal")
    kb, nb = _cells(df, clusters, df[col] == h["contrast_b"], "chose_focal")
    one = np.ones((1, len(clusters)))
    point = (_ratio(ka, na, one) - _ratio(kb, nb, one))[0]
    return 100 * point, 100 * (_ratio(ka, na, W) - _ratio(kb, nb, W)), int(na.sum() + nb.sum())


def est_focal_did(df, clusters, W, h):
    f1, f2 = f"cell_{h['factor']}", f"cell_{h['factor2']}"
    lv1 = [x.strip() for x in h["levels"].split(",")]
    base1, alt1 = h.get("control") or lv1[0], [x for x in lv1 if x != (h.get("control") or lv1[0])][0]
    def share(l1, l2, M):
        k, n = _cells(df, clusters, (df[f1] == l1) & (df[f2] == l2), "chose_focal")
        return _ratio(k, n, M), n.sum()
    one = np.ones((1, len(clusters)))
    parts = {}
    total = 0
    for l1 in (base1, alt1):
        for l2 in (h["contrast_a"], h["contrast_b"]):
            parts[(l1, l2, "p")], n = share(l1, l2, one)
            parts[(l1, l2, "b")], _ = share(l1, l2, W)
            total += n
    def did(t):
        return (parts[(alt1, h["contrast_a"], t)] - parts[(alt1, h["contrast_b"], t)]) - \
               (parts[(base1, h["contrast_a"], t)] - parts[(base1, h["contrast_b"], t)])
    return 100 * did("p")[0], 100 * did("b"), int(total)


def est_accuracy_vs_chance(df, clusters, W, h, n_rooms=4):
    k, n = _cells(df, clusters, df["block"].notna(), "chose_best")
    one = np.ones((1, len(clusters)))
    return 100 * (_ratio(k, n, one)[0] - 1 / n_rooms), 100 * (_ratio(k, n, W) - 1 / n_rooms), int(n.sum())


def est_share(col):
    def f(df, clusters, W, h):
        k, n = _cells(df, clusters, df["block"].notna(), col)
        one = np.ones((1, len(clusters)))
        return 100 * _ratio(k, n, one)[0], 100 * _ratio(k, n, W), int(n.sum())
    return f


def est_agreement_gap(df_all, clusters, W, h):
    """A_FS - mean(A_FF, A_SS) on trials matched by scenario, item, repeat and prompt (same layout)."""
    keys = ["scenario_id", "item", "repeat", "prompt_id"]
    d = df_all[["item_id", "cell_format", "retest", "chosen_room_uid"] + keys]
    def pick(fmt, rt):
        x = d[(d["cell_format"] == fmt) & (d["retest"] == rt)]
        return x.set_index(keys)[["item_id", "chosen_room_uid"]]
    F, S, F2, S2 = pick("full", False), pick("summary", False), pick("full", True), pick("summary", True)
    def agree(a, b):
        j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
        j["agree"] = (j["chosen_room_uid_a"] == j["chosen_room_uid_b"]).astype(float)
        g = j.groupby("item_id_a")["agree"]
        return (g.sum().reindex(clusters, fill_value=0).values, g.size().reindex(clusters, fill_value=0).values.astype(float))
    (kfs, nfs), (kff, nff), (kss, nss) = agree(F, S), agree(F, F2), agree(S, S2)
    one = np.ones((1, len(clusters)))
    def gap(M):
        return _ratio(kfs, nfs, M) - 0.5 * (_ratio(kff, nff, M) + _ratio(kss, nss, M))
    return 100 * gap(one)[0], 100 * gap(W), int(nfs.sum())


def est_best_omnibus(df, clusters, W, h):
    """Vector of (best-room share at each sort rule) - (share at control)."""
    col = f"cell_{h['factor']}"
    levels = [x.strip() for x in h["levels"].split(",")]
    ctrl = h.get("control") or levels[0]
    one = np.ones((1, len(clusters)))
    kc, nc = _cells(df, clusters, df[col] == ctrl, "chose_best")
    pts, boots, labels, total = [], [], [], nc.sum()
    for lv in levels:
        if lv == ctrl:
            continue
        k, n = _cells(df, clusters, df[col] == lv, "chose_best")
        total += n.sum()
        pts.append(100 * (_ratio(k, n, one)[0] - _ratio(kc, nc, one)[0]))
        boots.append(100 * (_ratio(k, n, W) - _ratio(kc, nc, W)))
        labels.append(lv)
    return np.array(pts), np.column_stack(boots), int(total), labels


ESTIMANDS: dict[str, Callable] = {
    "focal_diff": est_focal_diff, "focal_did": est_focal_did, "accuracy_vs_chance": est_accuracy_vs_chance,
    "threshold_best": est_share("chose_best"), "threshold_focal": est_share("chose_focal"),
    "agreement_gap": est_agreement_gap,
}


# ------------------------------------------------------------------ helpers
def _ci(b: np.ndarray, level: float) -> tuple[float, float]:
    b = b[~np.isnan(b)]
    if len(b) == 0:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    return float(np.percentile(b, 100 * a)), float(np.percentile(b, 100 * (1 - a)))


def _p(est: float, b: np.ndarray, G: int, one_sided: Optional[str] = None) -> float:
    b = b[~np.isnan(b)]
    se = float(np.std(b, ddof=1)) if len(b) > 1 else math.nan
    if not se or math.isnan(se) or math.isnan(est):
        return math.nan
    t = est / se
    df = max(G - 1, 1)
    if one_sided == "-":
        return float(stats.t.cdf(t, df))
    if one_sided == "+":
        return float(stats.t.sf(t, df))
    return float(2 * stats.t.sf(abs(t), df))


def benjamini_hochberg(p: list[float]) -> list[float]:
    arr = np.array([np.nan if v is None else v for v in p], dtype=float)
    q = np.full(len(arr), np.nan)
    idx = np.where(~np.isnan(arr))[0]
    if len(idx) == 0:
        return q.tolist()
    order = idx[np.argsort(arr[idx])]
    m = len(order)
    ranked = arr[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    q[order] = ranked
    return q.tolist()


def lakens_verdict(sig: bool, equiv: bool) -> str:
    if sig and not equiv:
        return "Effect"
    if sig and equiv:
        return "Real but negligible"
    if equiv:
        return "No meaningful effect"
    return "Inconclusive"


def plain_english(h: dict[str, Any], r: dict[str, Any]) -> str:
    hid, est = h["hypothesis_id"], r["estimate"]
    if math.isnan(est):
        return f"{hid}: not enough data yet."
    ci = f"95% CI {r['ci95_low']:+.1f} to {r['ci95_high']:+.1f}"
    fam = h.get("family")
    if fam == "check":
        return (f"{hid}: {r['estimate']:.1f}% ({ci}). Needs >= {float(h['pass_threshold']) * 100:.0f}% "
                f"-> {'PASS' if r['verdict'] == 'PASS' else 'FAIL'}.") if h["estimand"].startswith("threshold") else \
               f"{hid}: difference {est:+.1f} pp ({ci}), p = {r['p']:.3f} -> {r['verdict']}."
    if fam == "gate":
        return (f"{hid}: accuracy is {est + 25:.1f}% vs 25% chance ({est:+.1f} pp, {ci}); "
                f"top single room chosen {r.get('max_room_share', math.nan):.0f}% of the time -> {r['verdict']}.")
    unit = "pp of agreement" if h["estimand"] == "agreement_gap" else "pp"
    what = {"agreement_gap": "Full-vs-summary agreement minus rerun agreement",
            "best_omnibus": "Largest shift in the best room's share across sort rules",
            "focal_did": "Change in the complaint penalty (summary minus full)"}.get(
        h["estimand"], f"Focal room share with '{h['contrast_a']}' minus '{h['contrast_b']}'")
    dec = "Reject H0" if r["significant"] else "Do not reject H0"
    dirn = ""
    if r["significant"] and h.get("predicted") in ("+", "-"):
        dirn = " (as predicted)" if r["direction"] == h["predicted"] else " (opposite to prediction)"
    return (f"{hid}: {what} = {est:+.1f} {unit} ({ci}), p = {r['p']:.3f}"
            f"{', q = %.3f' % r['q_bh'] if not math.isnan(r.get('q_bh', math.nan)) else ''}. "
            f"{dec}{dirn}. Verdict: {r['verdict']}.")


# ------------------------------------------------------------------ main
def load_block_trials(study: str, block: str) -> Optional[pd.DataFrame]:
    p = RUNS_DIR / block_experiment(study, block) / "tables" / "trials.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    for c in ("retest", "parse_ok", "api_error"):
        if c in df:
            df[c] = df[c].astype(str).str.lower().isin(["true", "1"])
    df = df[df["parse_ok"] & ~df["api_error"]].copy()
    for c in ("chose_best", "chose_focal"):
        df[c] = df[c].astype(str).str.lower().isin(["true", "1"]).astype(float)
    return df


def evaluate(study: str, hypotheses: list[dict[str, Any]], settings: dict[str, Any],
             seed: int = 7) -> list[dict[str, Any]]:
    alpha = float(settings.get("alpha", 0.05))
    default_sesoi = float(settings.get("sesoi_pp", 5))
    B = int(settings.get("bootstrap_reps", 2000))
    rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for h in hypotheses:
        if not h.get("enabled", True):
            continue
        df_block = load_block_trials(study, h["block"])
        base = {"timestamp": now, "experiment": block_experiment(study, h["block"]), "hypothesis_id": h["hypothesis_id"],
                "family": h.get("family", ""), "estimand": h["estimand"], "predicted": h.get("predicted", ""),
                "sesoi": _sesoi_pp(h, default_sesoi)}
        if df_block is None or df_block.empty:
            rows.append({**base, "arm": "(all)", "verdict": "Not run yet", "estimate": math.nan,
                         "plain_english": f"{h['hypothesis_id']}: not run yet (block '{h['block']}')."})
            continue
        clusters = sorted(df_block["item_id"].unique())
        G = len(clusters)
        if G < 4:
            rows.append({**base, "arm": "(all)", "clusters": G, "verdict": "Too little data",
                         "estimate": math.nan, "n_trials": len(df_block),
                         "plain_english": f"{h['hypothesis_id']}: only {G} independent unit(s) (scenario x room) so far. "
                                          f"Run more trials (a pilot of 5+ per cell covers several scenarios)."})
            continue
        rng = np.random.default_rng(seed)
        W = _weights(G, B, rng)          # same resamples for every arm -> valid pooled estimate
        arms = sorted(df_block["arm"].unique())
        per_arm = {}
        for arm in arms:
            d_all = df_block[df_block["arm"] == arm]
            d = d_all if h["estimand"] == "agreement_gap" else d_all[~d_all["retest"]]
            if h["estimand"] == "best_omnibus":
                per_arm[arm] = est_best_omnibus(d, clusters, W, h)
            else:
                per_arm[arm] = ESTIMANDS[h["estimand"]](d, clusters, W, h)
        groups = [(a, [a]) for a in arms] + ([("(pooled: mean of models)", arms)] if len(arms) > 1 else [])
        for label, members in groups:
            r = {**base, "arm": label, "clusters": G, "unit": "pp"}
            if h["estimand"] == "best_omnibus":
                pts = np.mean([per_arm[a][0] for a in members], axis=0)
                boots = np.mean([per_arm[a][1] for a in members], axis=0)
                labels = per_arm[members[0]][3]
                n = sum(per_arm[a][2] for a in members)
                ok = ~np.isnan(boots).any(axis=1)
                cov = np.cov(boots[ok].T) if ok.sum() > 2 else None
                try:
                    wald = float(pts @ np.linalg.pinv(np.atleast_2d(cov)) @ pts) if cov is not None else math.nan
                    p = float(stats.chi2.sf(wald, len(pts)))
                except Exception:
                    p = math.nan
                i = int(np.nanargmax(np.abs(pts)))
                cis90 = [_ci(boots[:, j], 0.90) for j in range(len(pts))]
                lo95, hi95 = _ci(boots[:, i], 0.95)
                equiv = all(lo > -r["sesoi"] and hi < r["sesoi"] for lo, hi in cis90)
                r.update(estimate=float(pts[i]), ci95_low=lo95, ci95_high=hi95, ci90_low=cis90[i][0],
                         ci90_high=cis90[i][1], p=p, n_trials=n, equivalent=equiv,
                         notes="; ".join(f"{l}: {v:+.1f}" for l, v in zip(labels, pts)))
            else:
                est = float(np.mean([per_arm[a][0] for a in members]))
                boot = np.mean([per_arm[a][1] for a in members], axis=0)
                n = sum(per_arm[a][2] for a in members)
                one = "-" if h["estimand"] == "agreement_gap" else None
                lo95, hi95 = _ci(boot, 0.95)
                lo90, hi90 = _ci(boot, 0.90)
                r.update(estimate=est, ci95_low=lo95, ci95_high=hi95, ci90_low=lo90, ci90_high=hi90,
                         p=_p(est, boot, G, one), n_trials=n,
                         equivalent=bool(lo90 > -r["sesoi"] and hi90 < r["sesoi"]))
                if h["estimand"].startswith("threshold"):
                    r["unit"] = "% chosen"
            r["direction"] = "+" if r["estimate"] > 0 else "-" if r["estimate"] < 0 else "0"
            if h["estimand"] == "accuracy_vs_chance":
                d = df_block[(df_block["arm"].isin(members)) & ~df_block["retest"]]
                r["max_room_share"] = float(d["chosen_room_uid"].value_counts(normalize=True).max() * 100) if len(d) else math.nan
            rows.append(r)
        # heterogeneity across models (Cochran's Q with bootstrap SEs), on the per-arm rows
        if len(arms) > 1 and h["estimand"] != "best_omnibus":
            ests = np.array([per_arm[a][0] for a in arms])
            ses = np.array([np.nanstd(per_arm[a][1], ddof=1) for a in arms])
            w = 1 / np.where(ses > 0, ses, np.nan) ** 2
            if np.all(np.isfinite(w)):
                mu = np.sum(w * ests) / np.sum(w)
                Q = float(np.sum(w * (ests - mu) ** 2))
                hp = float(stats.chi2.sf(Q, len(arms) - 1))
                for r in rows:
                    if r["hypothesis_id"] == h["hypothesis_id"] and r.get("arm", "").startswith("(pooled"):
                        r["heterogeneity_p"] = hp

    # multiple testing: BH across confirmatory hypotheses, within each arm label
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("family") == "confirmatory" and not math.isnan(r.get("estimate", math.nan)):
            by_arm.setdefault(r["arm"], []).append(r)
    for arm, rs in by_arm.items():
        for r, q in zip(rs, benjamini_hochberg([r.get("p") for r in rs])):
            r["q_bh"] = q

    hmap = {h["hypothesis_id"]: h for h in hypotheses}
    for r in rows:
        if math.isnan(r.get("estimate", math.nan)):
            continue
        h = hmap[r["hypothesis_id"]]
        fam = h.get("family")
        if fam == "check" and h["estimand"].startswith("threshold"):
            thr = float(h.get("pass_threshold") or 0) * 100
            r["verdict"] = "PASS" if r["estimate"] >= thr else "FAIL"
            r["significant"], r["decision_H0"] = "", ""
        elif fam == "gate":
            sig = (r.get("p", 1) < alpha) and r["estimate"] > 0
            r["verdict"] = "PASS" if sig else "FAIL"
            r["significant"], r["decision_H0"] = sig, "Reject H0" if sig else "Do not reject H0"
        else:
            pv = r.get("q_bh") if fam == "confirmatory" and not math.isnan(r.get("q_bh", math.nan)) else r.get("p")
            sig = bool(pv is not None and not math.isnan(pv) and pv < alpha)
            r["significant"] = sig
            r["decision_H0"] = "Reject H0" if sig else "Do not reject H0"
            if fam == "check":   # placebo: pass when there is NO effect
                r["verdict"] = "PASS" if not sig else "FAIL"
            else:
                r["verdict"] = lakens_verdict(sig, bool(r.get("equivalent")))
        r["plain_english"] = plain_english(h, r)
    return rows
