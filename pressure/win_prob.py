"""Ball-level win-probability model, Win Probability Added and Leverage Index."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config as C

F1 = ["score", "wkts", "balls_rem", "max_balls", "par"]
MONO1 = [1, -1, 1, 0, -1]
F2 = ["runs_needed", "balls_rem", "wkts_in_hand", "rrr", "target"]
MONO2 = [-1, 1, 1, -1, 0]

OUTCOMES = {"0": (0, 1, 0), "1": (1, 1, 0), "2": (2, 1, 0), "4": (4, 1, 0),
            "6": (6, 1, 0), "W": (0, 1, 1), "X": (1, 0, 0)}  # runs, legal, wicket


def _clf(mono):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, monotonic_cst=mono, random_state=C.SEED)


def add_par(balls):
    """Par = mean completed first-innings total per (IPL vs other, year)."""
    inn1 = balls[balls.innings == 1].groupby("match_id").agg(
        total=("innings_total", "first"), maxb=("max_balls", "first"),
        is_ipl=("is_ipl", "first"), year=("year", "first"))
    par = inn1[inn1.maxb == C.FULL_BALLS].groupby(["is_ipl", "year"])["total"].mean().rename("par")
    out = balls.drop(columns="par", errors="ignore").merge(par.reset_index(), on=["is_ipl", "year"], how="left")
    out["par"] = out["par"].fillna(par.mean())
    return out


def _x1(score, wkts, balls_rem, max_balls, par):
    return np.column_stack([score, wkts, balls_rem, max_balls, par])


def _x2(runs_needed, balls_rem, wkts_in_hand, target):
    rrr = runs_needed * 6 / np.maximum(balls_rem, 1)
    return np.column_stack([runs_needed, balls_rem, wkts_in_hand, rrr, target])


def fit_models(train):
    t1 = train[train.innings == 1]
    t2 = train[train.innings == 2]
    m1 = _clf(MONO1).fit(_x1(t1.score_before, t1.wkts_before, t1.balls_rem, t1.max_balls, t1.par), t1.bat_won)
    m2 = _clf(MONO2).fit(_x2(t2.runs_needed, t2.balls_rem, t2.wkts_in_hand, t2.target), t2.bat_won)
    return m1, m2


def wp_state(models, innings, score, wkts, legal, max_balls, par, max2, target2_fixed):
    """WP for the batting team of `innings` at a (possibly terminal) state."""
    m1, m2 = models
    innings, score, wkts, legal = map(np.asarray, (innings, score, wkts, legal))
    max_balls, par, max2, target2_fixed = map(np.asarray, (max_balls, par, max2, target2_fixed))
    out = np.full(len(score), np.nan)
    done = (legal >= max_balls) | (wkts >= 10)

    i1 = (innings == 1) & ~done
    if i1.any():
        out[i1] = m1.predict_proba(_x1(score[i1], wkts[i1], max_balls[i1] - legal[i1], max_balls[i1], par[i1]))[:, 1]
    e1 = (innings == 1) & done
    if e1.any():
        # value innings-1 end with the innings-1 model (balls_rem=0) to avoid a model-switch jump
        n = e1.sum()
        out[e1] = m1.predict_proba(_x1(score[e1], np.minimum(wkts[e1], 9), np.zeros(n), max_balls[e1], par[e1]))[:, 1]

    i2 = innings == 2
    tgt2 = target2_fixed
    need = tgt2 - score
    won = i2 & (need <= 0)
    lost = i2 & ~won & done
    live = i2 & ~won & ~done
    out[won] = 1.0
    out[lost] = np.where(need[lost] == 1, 0.5, 0.0)
    if live.any():
        out[live] = m2.predict_proba(_x2(need[live], max_balls[live] - legal[live], 10 - wkts[live], tgt2[live]))[:, 1]
    return out


def outcome_probs(balls):
    """Empirical outcome distribution per (innings, phase, wicket band)."""
    b = balls
    cat = np.select(
        [~b.is_legal, b.team_wicket, b.total_runs >= 6, b.total_runs >= 4, b.total_runs >= 2, b.total_runs == 1],
        ["X", "W", "6", "4", "2", "1"], default="0")
    key = b[["innings", "phase"]].assign(wband=pd.cut(b.wkts_before, [-1, 2, 5, 10], labels=["0-2", "3-5", "6+"]).astype(str), cat=cat)
    p = key.groupby(["innings", "phase", "wband"])["cat"].value_counts(normalize=True).unstack(fill_value=0)
    return p.reindex(columns=list(OUTCOMES), fill_value=0)


def _match_cols(df):
    max2 = df["second_inn_revised_overs"].fillna(C.FULL_BALLS).to_numpy(float)
    t2 = df["revised_target"].to_numpy(float)
    # innings-2 balls: target is known; innings-1 terminal: use revised target if set else score+1
    target_inn2 = np.where(df.innings == 2, df["target"], t2)
    return max2, target_inn2


def score_fold(models, df, probs):
    max2, tgt = _match_cols(df)
    base = dict(max_balls=df.max_balls.to_numpy(float), par=df.par.to_numpy(float), max2=max2, target2_fixed=tgt)
    inn = df.innings.to_numpy()
    s0, w0, l0 = df.score_before.to_numpy(float), df.wkts_before.to_numpy(float), df.legal_before.to_numpy(float)

    wp0 = wp_state(models, inn, s0, w0, l0, **base)
    wp1 = wp_state(models, inn, s0 + df.total_runs, w0 + df.team_wicket, l0 + df.is_legal, **base)

    wband = pd.cut(df.wkts_before, [-1, 2, 5, 10], labels=["0-2", "3-5", "6+"]).astype(str)
    pk = probs.reindex(pd.MultiIndex.from_arrays([df.innings, df.phase, wband])).fillna(0).to_numpy()
    swing = np.zeros(len(df))
    for j, (r, lg, wk) in enumerate(OUTCOMES.values()):
        wpk = wp_state(models, inn, s0 + r, w0 + wk, l0 + lg, **base)
        swing += pk[:, j] * np.abs(wpk - wp0)
    return pd.DataFrame({"wp_before": wp0, "wp_after": wp1, "li_raw": swing}, index=df.index)


def _tied_matches(balls):
    last = balls[balls.innings == 2].groupby("match_id").agg(s=("innings_total", "first"), t=("target", "first"))
    return set(last.index[last.s == last.t - 1])


def cross_fit(balls, n_splits=5):
    """Out-of-fold WP/LI so no match is scored by a model that saw its outcome."""
    balls = add_par(balls)
    trainable = balls.has_winner & ~balls.match_id.isin(_tied_matches(balls))
    probs = outcome_probs(balls)
    res = []
    for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(balls, groups=balls.match_id):
        tr = balls.iloc[tr_idx]
        models = fit_models(tr[trainable.iloc[tr_idx].to_numpy()])
        res.append(score_fold(models, balls.iloc[te_idx], probs))
    out = balls.join(pd.concat(res))
    out["wpa_bat"] = out.wp_after - out.wp_before
    out["leverage"] = out.li_raw / out.li_raw.mean()
    return out, probs


def evaluate(balls_scored):
    d = balls_scored[balls_scored.bat_won.notna() & balls_scored.bat_won.isin([0, 1])]
    y, p = d.bat_won.to_numpy(), d.wp_before.clip(1e-4, 1 - 1e-4).to_numpy()
    rows = {"balls": len(d), "brier": brier_score_loss(y, p), "brier_coinflip": 0.25,
            "log_loss": log_loss(y, p), "auc": roc_auc_score(y, p)}
    for inn in (1, 2):
        s = d.innings == inn
        rows[f"brier_inn{inn}"] = brier_score_loss(y[s], p[s])
    bins = pd.cut(p, np.linspace(0, 1, 11), include_lowest=True)
    calib = pd.DataFrame({"pred": p, "obs": y}).groupby(bins, observed=True).agg(
        mean_pred=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size"))
    return rows, calib


def temporal_holdout(balls, test_year=C.LATEST_YEAR):
    """Train on earlier seasons, test on the latest season."""
    balls = add_par(balls).drop(columns=["wp_before", "wp_after", "li_raw", "wpa_bat", "leverage"], errors="ignore")
    ok = balls.has_winner & ~balls.match_id.isin(_tied_matches(balls))
    tr = balls[ok & (balls.year < test_year)]
    te = balls[ok & (balls.year == test_year)]
    models = fit_models(tr)
    probs = outcome_probs(tr)
    sc = te.join(score_fold(models, te, probs))
    return evaluate(sc)
