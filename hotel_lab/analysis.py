"""Pre-registered style analysis, following the brief's "How to analyse" steps.

1. Describe first: failure rates, then choice shares with 95% intervals.
2. Baseline: accuracy (share choosing the fact-sheet-best room) per cell and arm.
3. Effect: for focal blocks, the focal room's share under each level minus its share in the control
   cell, with a cluster bootstrap interval (clusters = items). Reported as "from X% to Y%".
4. Conditional logit per arm and block: room fixed effects (per scenario), shell fixed effects,
   position dummies; standard errors clustered by item; p-values from t(G-1) because the number of
   clusters G is small (Judd, Westfall & Kenny, 2012).
5. Summary vs full (H1): agreement and Cohen's kappa on matched trials (same item, repeat and
   layout), compared with the test-retest agreement of identical trials.
6. Benjamini-Hochberg across the tested coefficients.
An "arm" = model x interface x reasoning depth. Arms are always estimated separately.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from hotel_lab import RUNS_DIR

Z = 1.959964
CSV = {"index": False, "encoding": "utf-8-sig"}


def wilson(k: float, n: float) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / den
    return (centre - half, centre + half)


def load_design(experiment: str) -> dict[str, Any]:
    p = RUNS_DIR / experiment / "experiment.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def block_specs(design: dict[str, Any], present_blocks: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Enabled blocks that actually have data."""
    return {b["name"]: b["factors"] for b in design["blocks"]
            if b.get("enabled", True) and (present_blocks is None or b["name"] in present_blocks)}


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return all(c in df.columns for c in cols)


# --------------------------------------------------------------------------- describe
def failure_table(trials: pd.DataFrame) -> pd.DataFrame:
    t = trials.copy()
    t["not_usable"] = ~t["parse_ok"].astype(bool)
    t["first_attempt_failed"] = ~t["first_attempt_ok"].fillna(False).astype(bool) if "first_attempt_ok" in t else np.nan
    t["cut_off"] = t["finish_reason"].astype(str).eq("length") if "finish_reason" in t else False
    t["truncation"] = t["possible_truncation"].fillna(False).astype(bool) if "possible_truncation" in t else False
    t["mismatch"] = t["name_mismatch"].map(lambda v: v is True or v == "True") if "name_mismatch" in t else False
    out = t.groupby(["arm", "block", "cell"]).agg(
        n=("trial_id", "size"), api_errors=("api_error", "sum"), not_usable=("not_usable", "sum"),
        first_attempt_failed=("first_attempt_failed", "sum"), reply_cut_off=("cut_off", "sum"),
        prompt_maybe_truncated=("truncation", "sum"), hotel_name_mismatch=("mismatch", "sum")).reset_index()
    out["failure_rate"] = (out["not_usable"] / out["n"]).round(4)
    out["first_attempt_failure_rate"] = (out["first_attempt_failed"] / out["n"]).round(4)
    out["over_5pct"] = out["failure_rate"] > 0.05
    return out


def share_table(ok: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    rows = []
    for (m, b, c), d in ok.groupby(["arm", "block", "cell"]):
        k, n = int(d[col].sum()), len(d)
        lo, hi = wilson(k, n)
        rows.append({"arm": m, "block": b, "cell": c, "n": n, name: k / n if n else math.nan,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows, columns=["arm", "block", "cell", "n", name, "ci_low", "ci_high"])


def position_table(ok: pd.DataFrame) -> pd.DataFrame:
    t = ok.groupby(["arm", "chosen_position"]).size().rename("n").reset_index()
    t["share"] = t["n"] / t.groupby("arm")["n"].transform("sum")
    return t


def _bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, reps: int, rng: np.random.Generator):
    """Difference in means (a - b) with a bootstrap that resamples whole items (clusters)."""
    items = sorted(set(a["item_id"]) | set(b["item_id"]))
    ga = {i: d[col].astype(float).values for i, d in a.groupby("item_id")}
    gb = {i: d[col].astype(float).values for i, d in b.groupby("item_id")}
    diffs = []
    for _ in range(reps):
        pick = rng.choice(items, size=len(items), replace=True)
        va = np.concatenate([ga.get(i, np.array([])) for i in pick])
        vb = np.concatenate([gb.get(i, np.array([])) for i in pick])
        if len(va) and len(vb):
            diffs.append(va.mean() - vb.mean())
    if not diffs:
        return (math.nan, math.nan)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def focal_effects(ok: pd.DataFrame, specs: dict[str, dict], reps: int = 1000, seed: int = 1) -> pd.DataFrame:
    """Surface-sensitivity index: focal-room share at each level minus its share in the control cell."""
    rng = np.random.default_rng(seed)
    rows = []
    for block, factors in specs.items():
        if len(factors) != 1:
            continue  # interactions go through the logit
        fname, spec = next(iter(factors.items()))
        col = f"cell_{fname}"
        if spec["assign"] != "focal" or col not in ok:
            continue
        d = ok[ok["block"] == block]
        for m, dm in d.groupby("arm"):
            ctrl = dm[dm[col] == spec["control"]]
            base = ctrl["chose_focal"].astype(float).mean() if len(ctrl) else math.nan
            for lvl in spec["levels"]:
                if lvl == spec["control"]:
                    continue
                tr = dm[dm[col] == lvl]
                if tr.empty or ctrl.empty:
                    continue
                share = tr["chose_focal"].astype(float).mean()
                lo, hi = _bootstrap_diff(tr, ctrl, "chose_focal", reps, rng)
                rows.append({"arm": m, "block": block, "factor": fname, "level": lvl,
                             "control": spec["control"], "n_treated": len(tr), "n_control": len(ctrl),
                             "clusters": dm["item_id"].nunique(),
                             "focal_share_control": base, "focal_share_treated": share,
                             "diff_pp": 100 * (share - base), "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- H1: summary vs full
def cohen_kappa(a: list, b: list) -> float:
    labels = sorted(set(a) | set(b))
    if len(a) == 0:
        return math.nan
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        mat[idx[x], idx[y]] += 1
    n = mat.sum()
    po = np.trace(mat) / n
    pe = (mat.sum(0) * mat.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def _pairs(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = keys + ["chosen_room_uid", "chose_best"]
    return a[cols].merge(b[cols], on=keys, how="inner", suffixes=("_a", "_b"))


def _agree(p: pd.DataFrame) -> tuple[float, float, int]:
    if p.empty:
        return (math.nan, math.nan, 0)
    return (float((p["chosen_room_uid_a"] == p["chosen_room_uid_b"]).mean()),
            cohen_kappa(p["chosen_room_uid_a"].tolist(), p["chosen_room_uid_b"].tolist()), len(p))


def summary_vs_full(ok_all: pd.DataFrame, specs: dict[str, dict], reps: int = 1000) -> pd.DataFrame:
    """Matched trials share item, repeat and layout. Test-retest pairs are identical trials run twice:
    their agreement is the ceiling to compare full-vs-summary agreement against (temperature 1.0 is noisy)."""
    rng = np.random.default_rng(2)
    rows = []
    for block, factors in specs.items():
        spec = factors.get("format")
        if not spec or spec["assign"] != "whole_set" or not {"full", "summary"} <= set(spec["levels"]):
            continue
        if "cell_format" not in ok_all:
            continue
        other = [f"cell_{f}" for f in factors if f != "format" and f"cell_{f}" in ok_all]
        keys = ["arm", "scenario_id", "item", "repeat", "prompt_id"] + other
        d = ok_all[ok_all["block"] == block]
        main, rt = d[~d["retest"]], d[d["retest"]]
        full, summ = main[main["cell_format"] == "full"], main[main["cell_format"] == "summary"]
        for m in sorted(d["arm"].unique()):
            f_m, s_m = full[full["arm"] == m], summ[summ["arm"] == m]
            a_fs, k_fs, n_fs = _agree(_pairs(f_m, s_m, keys))
            a_ff, k_ff, n_ff = _agree(_pairs(f_m, rt[(rt["arm"] == m) & (rt["cell_format"] == "full")], keys))
            a_ss, k_ss, n_ss = _agree(_pairs(s_m, rt[(rt["arm"] == m) & (rt["cell_format"] == "summary")], keys))
            lo, hi = _bootstrap_diff(s_m, f_m, "chose_best", reps, rng) if len(f_m) and len(s_m) else (math.nan,) * 2
            p = _pairs(f_m, s_m, keys)
            rows.append({
                "arm": m, "block": block,
                "pairs_full_vs_summary": n_fs, "agreement_full_vs_summary": a_fs, "kappa_full_vs_summary": k_fs,
                "pairs_full_retest": n_ff, "agreement_full_vs_full": a_ff, "kappa_full_vs_full": k_ff,
                "pairs_summary_retest": n_ss, "agreement_summary_vs_summary": a_ss, "kappa_summary_vs_summary": k_ss,
                "agreement_gap_vs_retest": a_fs - np.nanmean([a_ff, a_ss]) if n_ff or n_ss else math.nan,
                "best_share_full": f_m["chose_best"].astype(float).mean(),
                "best_share_summary": s_m["chose_best"].astype(float).mean(),
                "best_share_diff_pp": 100 * (s_m["chose_best"].astype(float).mean() - f_m["chose_best"].astype(float).mean()),
                "diff_ci_low_pp": 100 * lo, "diff_ci_high_pp": 100 * hi,
                "disagree_toward_best": int(((~p["chose_best_a"].astype(bool)) & p["chose_best_b"].astype(bool)).sum()),
                "disagree_away_from_best": int((p["chose_best_a"].astype(bool) & ~p["chose_best_b"].astype(bool)).sum()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- conditional logit
def _clogit_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray, clusters: np.ndarray, iters: int = 60):
    """Conditional logit by Newton-Raphson, with cluster-robust (sandwich) standard errors."""
    beta = np.zeros(X.shape[1])
    order = np.argsort(groups, kind="stable")
    y, X, groups, clusters = y[order], X[order], groups[order], clusters[order]
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    spans = list(zip(starts[:-1], starts[1:]))
    H = np.zeros((len(beta), len(beta)))
    for _ in range(iters):
        grad = np.zeros_like(beta)
        H = np.zeros((len(beta), len(beta)))
        for s, e in spans:
            Xg, yg = X[s:e], y[s:e]
            u = Xg @ beta
            p = np.exp(u - u.max())
            p /= p.sum()
            xbar = p @ Xg
            grad += yg @ Xg - xbar
            Xc = Xg - xbar
            H -= (Xc * p[:, None]).T @ Xc
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
        step = np.clip(step, -5, 5)        # damped step: stops a never-chosen room from blowing up
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    scores = {}
    for s, e in spans:
        Xg, yg = X[s:e], y[s:e]
        u = Xg @ beta
        p = np.exp(u - u.max())
        p /= p.sum()
        scores[clusters[s]] = scores.get(clusters[s], 0) + (yg @ Xg - p @ Xg)
    S = np.array(list(scores.values()))
    G = len(S)
    Hinv = np.linalg.pinv(-H)
    V = Hinv @ (S.T @ S * (G / (G - 1) if G > 1 else 1)) @ Hinv
    return beta, np.sqrt(np.clip(np.diag(V), 0, None)), G


def _fe_dummies(d: pd.DataFrame, col: str, prefix: str) -> dict[str, np.ndarray]:
    """Fixed-effect dummies with one reference level PER SCENARIO (rooms only compete within a scenario)."""
    out = {}
    for _, ds in d.groupby("scenario_id"):
        levels = sorted(ds[col].unique())
        for lv in levels[1:]:
            out[f"{prefix}[{lv}]"] = (d[col] == lv).astype(float).values
    return out


def _within_variation(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """True for columns that vary inside at least one choice set (others cannot be identified)."""
    df = pd.DataFrame(X)
    dev = df - df.groupby(groups).transform("mean")
    return (dev.abs().max(axis=0) > 1e-12).values


def clogit_table(long: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:
    """Per arm x block: chosen ~ factor terms (+ interactions) + position + room FE + shell FE."""
    rows = []
    for block, factors in specs.items():
        d0 = long[(long["block"] == block) & (~long["retest"].astype(bool))]
        if d0.empty:
            continue
        for m, d in d0.groupby("arm"):
            d = d.reset_index(drop=True)
            terms: dict[str, np.ndarray] = {}
            tested: list[str] = []
            inds: dict[str, list[tuple[str, np.ndarray, str]]] = {}
            for fname, spec in factors.items():
                ref = spec.get("control", spec["levels"][0])
                inds[fname] = []
                for lvl in spec["levels"]:
                    if lvl == ref:
                        continue
                    ind = (d[fname] == lvl).astype(float).values
                    inds[fname].append((lvl, ind, spec["assign"]))
                    if spec["assign"] == "whole_set":
                        # constant within a choice set -> identified via its effect on the fact-best room
                        name = f"best_x_{fname}={lvl}"
                        terms[name] = ind * d["is_best"].values
                    else:
                        name = f"{fname}={lvl}"
                        terms[name] = ind
                    tested.append(name)
            # interactions between factors of the same block (e.g. format x review_set)
            fl = list(inds)
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    for la, ia, aa in inds[fl[i]]:
                        for lb, ib, ab in inds[fl[j]]:
                            name = f"{fl[i]}={la} x {fl[j]}={lb}"
                            col = ia * ib
                            if aa == "whole_set" and ab == "whole_set":
                                col = col * d["is_best"].values
                                name = "best_x_" + name
                            terms[name] = col
                            tested.append(name)
            if "order" in factors and d["first_review_rating"].notna().all():
                # H2 mechanism: does the star rating of the review shown FIRST move the room's choice?
                terms["first_review_rating"] = d["first_review_rating"].astype(float).values
                tested.append("first_review_rating")
            for pos in sorted(d["position"].unique())[1:]:
                terms[f"position_{pos}"] = (d["position"] == pos).astype(float).values
            terms.update(_fe_dummies(d, "room_uid", "room"))
            if "shell_uid" in d and (d["shell_uid"] != d["room_uid"]).any():
                terms.update(_fe_dummies(d, "shell_uid", "shell"))
            names = list(terms)
            X = np.column_stack([terms[n] for n in names])
            groups = d["choice_set_id"].values
            keep = _within_variation(X, groups)
            dropped = [n for n, k in zip(names, keep) if not k and n in tested]
            X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
            try:
                beta, se, G = _clogit_cluster(d["chosen"].values.astype(float), X, groups, d["item_id"].values)
            except Exception as e:
                rows.append({"arm": m, "block": block, "term": "ERROR", "note": str(e)})
                continue
            df = max(G - 1, 1)
            tcrit = stats.t.ppf(0.975, df)
            n_sets = d["choice_set_id"].nunique()
            for n_, b_, s_ in zip(names, beta, se):
                if n_.startswith(("room[", "shell[")):
                    continue
                t = b_ / s_ if s_ > 0 else math.nan
                p = 2 * stats.t.sf(abs(t), df) if s_ > 0 else math.nan
                rows.append({"arm": m, "block": block, "term": n_, "coef": b_, "se_cluster": s_,
                             "ci_low": b_ - tcrit * s_, "ci_high": b_ + tcrit * s_,
                             "odds_ratio": math.exp(b_) if abs(b_) < 50 else math.nan, "t": t, "df": df, "p": p,
                             "tested": n_ in tested, "choice_sets": n_sets, "clusters": G,
                             "note": "few clusters: treat p-values with caution" if G < 20 else ""})
            for n_ in dropped:
                rows.append({"arm": m, "block": block, "term": n_, "tested": False,
                             "note": "not identified (no variation within choice sets)"})
    return pd.DataFrame(rows)


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    mask = p.notna()
    q = pd.Series(np.nan, index=p.index)
    ps = p[mask].sort_values()
    m = len(ps)
    if m == 0:
        return q
    ranked = ps * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1].clip(upper=1)
    q[ps.index] = ranked.values
    return q


# --------------------------------------------------------------------------- figures
def plot_shares(df: pd.DataFrame, value: str, title: str, path: Path) -> None:
    if df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blocks = list(df["block"].unique())
    fig, axes = plt.subplots(len(blocks), 1, figsize=(9, 2.8 * len(blocks)), squeeze=False)
    for ax, b in zip(axes[:, 0], blocks):
        d = df[df["block"] == b]
        arms = list(d["arm"].unique())
        cells = list(d["cell"].unique())
        w = 0.8 / max(len(arms), 1)
        for i, m in enumerate(arms):
            dm = d[d["arm"] == m].set_index("cell").reindex(cells)
            x = np.arange(len(cells)) + i * w
            yerr = [dm[value] - dm["ci_low"], dm["ci_high"] - dm[value]]
            ax.bar(x, dm[value], w, yerr=yerr, capsize=3, label=m)
        ax.set_xticks(np.arange(len(cells)) + w * (len(arms) - 1) / 2)
        ax.set_xticklabels(cells, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{b}", fontsize=10)
        ax.set_ylabel(value)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def analyze(experiment: str, bootstrap_reps: int = 1000) -> Path:
    tables = RUNS_DIR / experiment / "tables"
    if not (tables / "trials.csv").exists():
        raise SystemExit("No tables yet. Run:  python -m hotel_lab export")
    trials = pd.read_csv(tables / "trials.csv")
    long = pd.read_csv(tables / "long.csv")
    if trials.empty:
        raise SystemExit("trials.csv is empty: no trials were logged.")
    for c in ("retest", "parse_ok", "api_error"):
        trials[c] = trials[c].astype(bool)
    if not long.empty:
        long["retest"] = long["retest"].astype(bool)
    specs = block_specs(load_design(experiment), set(trials["block"].unique()))
    out = RUNS_DIR / experiment / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    failure_table(trials).to_csv(out / "1_failure_rates.csv", **CSV)

    ok_all = trials[trials["parse_ok"] & ~trials["api_error"]].copy()
    if ok_all.empty:
        print("No usable trials (all failed to parse or errored). See 1_failure_rates.csv and the raw logs.")
        return out
    ok_all["chose_best"] = ok_all["chose_best"].astype(bool)
    ok_all["chose_focal"] = ok_all["chose_focal"].astype(bool)
    ok = ok_all[~ok_all["retest"]]

    acc = share_table(ok, "chose_best", "best_share")
    acc.to_csv(out / "2_accuracy_by_cell.csv", **CSV)
    foc = share_table(ok, "chose_focal", "focal_share")
    foc.to_csv(out / "3_focal_share_by_cell.csv", **CSV)
    position_table(ok).to_csv(out / "4_position_shares.csv", **CSV)
    focal_effects(ok, specs, reps=bootstrap_reps).to_csv(out / "5_surface_sensitivity.csv", **CSV)
    summary_vs_full(ok_all, specs, reps=bootstrap_reps).to_csv(out / "6_summary_vs_full.csv", **CSV)

    cl = clogit_table(long, specs) if not long.empty else pd.DataFrame()
    if not cl.empty and "tested" in cl:
        cl["q_bh"] = np.nan
        tested = cl["tested"].fillna(False).astype(bool) & cl["p"].notna()
        cl.loc[tested, "q_bh"] = benjamini_hochberg(cl.loc[tested, "p"])
    cl.to_csv(out / "7_conditional_logit.csv", **CSV)

    plot_shares(acc, "best_share", "Share choosing the fact-sheet-best room (95% CI)", out / "fig_accuracy.png")
    focal_blocks = [b for b, f in specs.items() if any(s["assign"] == "focal" for s in f.values())]
    plot_shares(foc[foc["block"].isin(focal_blocks)], "focal_share", "Share choosing the focal room (95% CI)",
                out / "fig_focal_share.png")

    with open(out / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "arm = model x interface x reasoning depth. Every table is per arm.\n\n"
            "1_failure_rates.csv      not-usable replies, first-attempt failures, cut-off replies, possibly\n"
            "                         truncated prompts, hotel-name mismatches (brief: keep failures under 5%)\n"
            "2_accuracy_by_cell.csv   share choosing the fact-sheet-best room, Wilson 95% CI\n"
            "3_focal_share_by_cell    share choosing the focal (treated) room\n"
            "4_position_shares.csv    share of choices by display position (position bias check)\n"
            "5_surface_sensitivity    focal share at each level minus control, in percentage points,\n"
            "                         95% CI from a bootstrap that resamples items (scenario x item)\n"
            "6_summary_vs_full.csv    H1. Agreement/kappa full vs summary on matched trials, next to the\n"
            "                         test-retest agreement of identical trials (the noise ceiling).\n"
            "                         A real format effect = full-vs-summary agreement clearly BELOW retest\n"
            "                         agreement, and/or a best-share difference whose CI excludes 0.\n"
            "7_conditional_logit.csv  per arm x block; room FE + shell FE (one reference per scenario) +\n"
            "                         position dummies; SE clustered by item; t(G-1) p-values; q_bh =\n"
            "                         Benjamini-Hochberg across tested terms. 'best_x_...' terms = how a\n"
            "                         whole-page change shifts the fact-best room's advantage.\n"
            "                         'first_review_rating' (order blocks) = effect of the first review's stars.\n"
            "With few clusters (G < 20) p-values are approximate; add scenarios to fix this.\n"
            "Everything not in the pre-registration must be labelled exploratory.\n"
        )
    return out
