"""Pre-registered style analysis, following the brief's "How to analyse" steps.

1. Describe first: choice shares with 95% intervals (failure rates checked before anything else).
2. Baseline: accuracy (share choosing the fact-sheet-best room) per cell and model.
3. Effect: for focal blocks, the focal room's share under each level minus its share in the control
   cell, with a cluster bootstrap interval (clusters = items). Reported as "from X% to Y%".
4. Conditional logit per model and block, with room fixed effects and position dummies,
   standard errors clustered by item (Judd, Westfall & Kenny, 2012).
5. Summary vs full: Cohen's kappa on matched trials, and the direction of disagreements.
6. Benjamini-Hochberg across the tested coefficients.
Models are always estimated separately (no blind averaging).
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


def block_specs(design: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {b["name"]: b["factors"] for b in design["blocks"]}


# --------------------------------------------------------------------------- shares
def failure_table(trials: pd.DataFrame) -> pd.DataFrame:
    g = trials.groupby(["model_name", "block", "cell"])
    out = g.agg(n=("trial_id", "size"), api_errors=("api_error", "sum"),
                parse_failures=("parse_ok", lambda s: int((~s.astype(bool)).sum()))).reset_index()
    out["failure_rate"] = (out["parse_failures"] / out["n"]).round(4)
    out["over_5pct"] = out["failure_rate"] > 0.05
    return out


def share_table(ok: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    rows = []
    for (m, b, c), d in ok.groupby(["model_name", "block", "cell"]):
        k, n = int(d[col].sum()), len(d)
        lo, hi = wilson(k, n)
        rows.append({"model_name": m, "block": b, "cell": c, "n": n, name: k / n if n else math.nan,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def position_table(ok: pd.DataFrame) -> pd.DataFrame:
    t = ok.groupby(["model_name", "chosen_position"]).size().rename("n").reset_index()
    t["share"] = t["n"] / t.groupby("model_name")["n"].transform("sum")
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
        focal = {f: s for f, s in factors.items() if s["assign"] == "focal"}
        if len(focal) != 1 or len(factors) != 1:
            continue  # single-factor focal blocks only; interactions go through the logit
        fname, spec = next(iter(focal.items()))
        d = ok[ok["block"] == block]
        if d.empty:
            continue
        col = f"cell_{fname}"
        for m, dm in d.groupby("model_name"):
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
                rows.append({"model_name": m, "block": block, "factor": fname, "level": lvl,
                             "control": spec["control"], "n_treated": len(tr), "n_control": len(ctrl),
                             "focal_share_control": base, "focal_share_treated": share,
                             "diff_pp": 100 * (share - base), "ci_low_pp": 100 * lo, "ci_high_pp": 100 * hi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- kappa
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


def summary_vs_full(ok: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for block, factors in specs.items():
        spec = factors.get("format")
        if not spec or spec["assign"] != "whole_set" or not {"full", "summary"} <= set(spec["levels"]):
            continue
        d = ok[ok["block"] == block]
        other = [f"cell_{f}" for f in factors if f != "format"]
        keys = ["model_name", "scenario_id", "item", "repeat", "prompt_id"] + other
        full = d[d["cell_format"] == "full"].set_index(keys)
        summ = d[d["cell_format"] == "summary"].set_index(keys)
        both = full[["chosen_room_id", "chose_best"]].join(summ[["chosen_room_id", "chose_best"]],
                                                           lsuffix="_full", rsuffix="_summary", how="inner")
        both = both.reset_index()
        for m, dm in both.groupby("model_name"):
            agree = (dm["chosen_room_id_full"] == dm["chosen_room_id_summary"]).mean()
            rows.append({
                "model_name": m, "block": block, "pairs": len(dm), "agreement": agree,
                "kappa": cohen_kappa(dm["chosen_room_id_full"].tolist(), dm["chosen_room_id_summary"].tolist()),
                "best_share_full": dm["chose_best_full"].astype(float).mean(),
                "best_share_summary": dm["chose_best_summary"].astype(float).mean(),
                "disagree_toward_best": int(((~dm["chose_best_full"].astype(bool)) & dm["chose_best_summary"].astype(bool)).sum()),
                "disagree_away_from_best": int((dm["chose_best_full"].astype(bool) & ~dm["chose_best_summary"].astype(bool)).sum()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- conditional logit
def _clogit_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray, clusters: np.ndarray, iters: int = 50):
    """Conditional logit by Newton-Raphson, with cluster-robust (sandwich) standard errors."""
    beta = np.zeros(X.shape[1])
    order = np.argsort(groups, kind="stable")
    y, X, groups, clusters = y[order], X[order], groups[order], clusters[order]
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    spans = list(zip(starts[:-1], starts[1:]))
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
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    # sandwich
    scores = {}
    for s, e in spans:
        Xg, yg = X[s:e], y[s:e]
        u = Xg @ beta
        p = np.exp(u - u.max())
        p /= p.sum()
        sc = yg @ Xg - p @ Xg
        c = clusters[s]
        scores[c] = scores.get(c, 0) + sc
    S = np.array(list(scores.values()))
    G = len(S)
    Hinv = np.linalg.pinv(-H)
    meat = S.T @ S * (G / (G - 1) if G > 1 else 1)
    V = Hinv @ meat @ Hinv
    return beta, np.sqrt(np.clip(np.diag(V), 0, None)), G


def clogit_table(long: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:
    """Per model x block: chosen ~ factor terms + position dummies + room fixed effects."""
    rows = []
    for block, factors in specs.items():
        d0 = long[long["block"] == block]
        if d0.empty:
            continue
        for m, d in d0.groupby("model_name"):
            d = d.copy()
            terms: dict[str, np.ndarray] = {}
            tested = []
            for fname, spec in factors.items():
                ref = spec.get("control", spec["levels"][0])
                for lvl in spec["levels"]:
                    if lvl == ref:
                        continue
                    ind = (d[fname] == lvl).astype(float).values
                    if spec["assign"] == "whole_set":
                        # constant within a choice set -> identify via its effect on the fact-best room
                        name = f"best_x_{fname}={lvl}"
                        terms[name] = ind * d["is_best"].values
                    else:
                        name = f"{fname}={lvl}"
                        terms[name] = ind
                    tested.append(name)
            for pos in sorted(d["position"].unique())[1:]:
                terms[f"position_{pos}"] = (d["position"] == pos).astype(float).values
            rooms = sorted(d["room_uid"].unique())
            for r in rooms[1:]:  # room fixed effects (one reference room)
                terms[f"room[{r}]"] = (d["room_uid"] == r).astype(float).values
            names = list(terms)
            X = np.column_stack([terms[n] for n in names])
            keep = X.std(axis=0) > 0
            X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
            try:
                beta, se, G = _clogit_cluster(d["chosen"].values.astype(float), X, d["choice_set_id"].values,
                                              d["item_id"].values)
            except Exception as e:
                rows.append({"model_name": m, "block": block, "term": "ERROR", "note": str(e)})
                continue
            n_sets = d["choice_set_id"].nunique()
            for n_, b_, s_ in zip(names, beta, se):
                if n_.startswith("room["):
                    continue
                z = b_ / s_ if s_ > 0 else math.nan
                p = 2 * (1 - stats.norm.cdf(abs(z))) if s_ > 0 else math.nan
                rows.append({"model_name": m, "block": block, "term": n_, "coef": b_, "se_cluster": s_,
                             "odds_ratio": math.exp(b_) if abs(b_) < 50 else math.nan, "z": z, "p": p,
                             "tested": n_ in tested, "choice_sets": n_sets, "clusters": G})
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
        models = list(d["model_name"].unique())
        cells = list(d["cell"].unique())
        w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            dm = d[d["model_name"] == m].set_index("cell").reindex(cells)
            x = np.arange(len(cells)) + i * w
            yerr = [dm[value] - dm["ci_low"], dm["ci_high"] - dm[value]]
            ax.bar(x, dm[value], w, yerr=yerr, capsize=3, label=m)
        ax.set_xticks(np.arange(len(cells)) + w * (len(models) - 1) / 2)
        ax.set_xticklabels(cells, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{b}", fontsize=10)
        ax.set_ylabel(value)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def analyze(experiment: str, bootstrap_reps: int = 1000) -> Path:
    tables = RUNS_DIR / experiment / "tables"
    trials = pd.read_csv(tables / "trials.csv")
    long = pd.read_csv(tables / "long.csv")
    specs = block_specs(load_design(experiment))
    out = RUNS_DIR / experiment / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    fails = failure_table(trials)
    fails.to_csv(out / "1_failure_rates.csv", index=False)

    ok = trials[(trials["parse_ok"]) & (~trials["api_error"])].copy()
    ok["chose_best"] = ok["chose_best"].astype(bool)
    ok["chose_focal"] = ok["chose_focal"].astype(bool)
    acc = share_table(ok, "chose_best", "best_share")
    acc.to_csv(out / "2_accuracy_by_cell.csv", index=False)
    foc = share_table(ok, "chose_focal", "focal_share")
    foc.to_csv(out / "3_focal_share_by_cell.csv", index=False)
    position_table(ok).to_csv(out / "4_position_shares.csv", index=False)

    eff = focal_effects(ok, specs, reps=bootstrap_reps)
    eff.to_csv(out / "5_surface_sensitivity.csv", index=False)

    svf = summary_vs_full(ok, specs)
    svf.to_csv(out / "6_summary_vs_full.csv", index=False)

    cl = clogit_table(long, specs)
    if not cl.empty and "tested" in cl:
        cl["q_bh"] = np.nan
        tested = cl["tested"].fillna(False).astype(bool)
        cl.loc[tested, "q_bh"] = benjamini_hochberg(cl.loc[tested, "p"])
    cl.to_csv(out / "7_conditional_logit.csv", index=False)

    plot_shares(acc, "best_share", "Share choosing the fact-sheet-best room (95% CI)", out / "fig_accuracy.png")
    focal_blocks = [b for b, f in specs.items() if any(s["assign"] == "focal" for s in f.values())]
    plot_shares(foc[foc["block"].isin(focal_blocks)], "focal_share", "Share choosing the focal room (95% CI)",
                out / "fig_focal_share.png")

    with open(out / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "1_failure_rates.csv      parse/API failures per model x cell (brief: keep under 5%)\n"
            "2_accuracy_by_cell.csv   share choosing the fact-sheet-best room, Wilson 95% CI\n"
            "3_focal_share_by_cell    share choosing the focal (treated) room\n"
            "4_position_shares.csv    share of choices by display position (position bias check)\n"
            "5_surface_sensitivity    focal share at each level minus control, in percentage points,\n"
            "                         95% CI from a bootstrap that resamples items (scenario x item)\n"
            "6_summary_vs_full.csv    matched trials: agreement, Cohen's kappa, direction of disagreement\n"
            "7_conditional_logit.csv  per model x block; room fixed effects + position dummies;\n"
            "                         SE clustered by item; q_bh = Benjamini-Hochberg across tested terms.\n"
            "                         'best_x_format=summary' = how a whole-page change shifts the\n"
            "                         fact-best room's advantage (whole-page factors cannot be estimated\n"
            "                         directly in a conditional logit).\n"
            "Everything not in the pre-registration must be labelled exploratory.\n"
        )
    return out
