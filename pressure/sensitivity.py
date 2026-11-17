"""Rank stability across alternative pressure definitions, thresholds and weightings."""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from . import config as C
from .metrics import player_table

GRID = {
    "definition": ["li", "rule"],
    "pctl": [0.75, 0.80, 0.85],
    "min_balls": [75, 100, 150],
    "weight_mode": ["full", "comp_only"],
    "weights": {
        "base": C.COMPOSITE_WEIGHTS,
        "impact_heavy": {"impact": 0.45, "skill": 0.20, "delta": 0.15, "consistency": 0.20},
        "skill_heavy": {"impact": 0.20, "skill": 0.40, "delta": 0.20, "consistency": 0.20},
    },
}


def run_grid(b, role, top=10):
    b = b.copy()
    records, ranks = [], {}
    combos = itertools.product(GRID["definition"], GRID["pctl"], GRID["min_balls"],
                               GRID["weight_mode"], GRID["weights"].items())
    for definition, pctl, mb, wm, (wname, w) in combos:
        b["flag"] = b[f"{definition}_pct"] >= pctl
        tab, _ = player_table(b, role, flag="flag", min_balls=mb, weight_mode=wm, weights=w)
        sid = f"{definition}|{pctl}|{mb}|{wm}|{wname}"
        ranks[sid] = tab["rank"]
        records.append({"scenario": sid, "eligible": len(tab)})
    R = pd.DataFrame(ranks)
    # ineligible in a scenario = ranked just below the last eligible player
    Rf = R.fillna(R.count() + 1)
    summary = pd.DataFrame({
        "scenarios_eligible": R.notna().sum(1),
        "median_rank": Rf.median(1),
        "mean_rank": Rf.mean(1),
        "top10_rate": (R <= top).sum(1) / R.shape[1],
        "best_rank": R.min(1),
        "worst_rank": R.max(1),
    }).sort_values(["median_rank", "mean_rank"])
    base = f"li|{C.PRESSURE_PCTL}|{C.MIN_PRESSURE_BALLS_BAT if role == 'bat' else C.MIN_PRESSURE_BALLS_BOWL}|full|base"
    taus = {c: kendalltau(R[base], R[c], nan_policy="omit").statistic for c in R.columns if c != base}
    return summary, R, pd.DataFrame(records), pd.Series(taus, name="kendall_tau_vs_base")


def one_at_a_time(taus, base):
    """Median Kendall tau for scenarios differing from base in exactly one dimension."""
    dims = ["definition", "pctl", "min_balls", "weight_mode", "weights"]
    b = base.split("|")
    rows = []
    for sid, tau in taus.items():
        diff = [d for d, x, y in zip(dims, sid.split("|"), b) if x != y]
        if len(diff) == 1:
            rows.append({"changed": diff[0], "scenario": sid, "kendall_tau": tau})
    return pd.DataFrame(rows)
