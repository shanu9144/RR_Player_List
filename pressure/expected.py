"""Context-expected runs and wickets per ball (out-of-fold), and wicket run value."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import config as C

FEATS = ["innings", "over_num", "wkts_before", "balls_rem", "rrr_f", "crr", "is_spin", "is_lhb", "is_ipl", "par"]


def _X(df):
    return df.assign(rrr_f=df.rrr.fillna(-1)).loc[:, FEATS].astype(float).to_numpy()


def _oof(df, mask, y, make_model, proba):
    pred = pd.Series(np.nan, index=df.index)
    sub = df[mask]
    for tr, te in GroupKFold(5).split(sub, groups=sub.match_id):
        m = make_model().fit(_X(sub.iloc[tr]), y[mask].iloc[tr])
        Xte = _X(sub.iloc[te])
        pred.iloc[np.flatnonzero(mask)[te]] = m.predict_proba(Xte)[:, 1] if proba else m.predict(Xte)
    return pred


def _reg():
    return HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05,
                                         min_samples_leaf=300, random_state=C.SEED)


def _clf():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=300, random_state=C.SEED)


def wicket_value_by_phase(b):
    """Runs a wicket costs: drop in expected remaining first-innings runs per extra wicket lost."""
    i1 = b[(b.innings == 1) & (b.max_balls == C.FULL_BALLS)].copy()
    i1["rem_runs"] = i1.innings_total - i1.score_before
    i1["wk"] = i1.wkts_before.clip(upper=8)
    tab = i1.groupby(["over_num", "wk"]).agg(r=("rem_runs", "mean"), n=("rem_runs", "size")).reset_index()
    tab = tab[tab.n >= 30]
    tab["r_next"] = tab.groupby("over_num").r.shift(-1)
    tab["cost"] = (tab.r - tab.r_next).clip(lower=0)
    tab = tab.dropna(subset=["cost"])
    tab["phase"] = pd.cut(tab.over_num, [0, 6, 15, 20], labels=["powerplay", "middle", "death"]).astype(str)
    return tab.groupby("phase").apply(lambda t: np.average(t.cost, weights=t.n), include_groups=False)


def add_expected(b):
    b = b.copy()
    faced = b.faced.to_numpy()
    allb = np.ones(len(b), bool)
    b["x_bat_runs"] = _oof(b, faced, b.runs.astype(float), _reg, False)
    b["x_bat_out"] = _oof(b, faced, b.striker_out.astype(int), _clf, True)
    b["x_bowl_runs"] = _oof(b, allb, b.bowler_runs.astype(float), _reg, False)
    b["x_bowl_wkt"] = _oof(b, allb, b.bowler_wicket.astype(int), _clf, True)

    lam = wicket_value_by_phase(b)
    b["wicket_value"] = b.phase.map(lam).astype(float)
    b["rv_bat"] = np.where(
        faced, (b.runs - b.x_bat_runs) - b.wicket_value * (b.striker_out.astype(int) - b.x_bat_out), np.nan)
    b["rv_bowl"] = (b.x_bowl_runs - b.bowler_runs) + b.wicket_value * (b.bowler_wicket.astype(int) - b.x_bowl_wkt)
    return b, lam
