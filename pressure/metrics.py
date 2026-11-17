"""Player-level pressure metrics with empirical-Bayes shrinkage and match-level consistency."""
import numpy as np
import pandas as pd

from . import config as C

ROLE = {
    "bat": dict(pid="striker_id", name="striker_name", team="batting_team_id", rv="rv_bat", wpa_sign=1),
    "bowl": dict(pid="bowler_id", name="bowler_name", team="bowling_team_id", rv="rv_bowl", wpa_sign=-1),
}


def eb_shrink(sum_x, sum_w, sum_x2, min_w=20):
    """Shrink weighted per-ball means toward the pooled mean; returns (shrunk, k)."""
    mean = sum_x / sum_w
    mu = sum_x.sum() / sum_w.sum()
    sigma2 = max(sum_x2.sum() / sum_w.sum() - mu ** 2, 1e-12)
    ok = sum_w >= min_w
    obs_var = np.average((mean[ok] - mu) ** 2, weights=sum_w[ok])
    tau2 = max(obs_var - sigma2 / np.average(sum_w[ok], weights=sum_w[ok]), sigma2 / 1000)
    k = sigma2 / tau2
    return (sum_x + k * mu) / (sum_w + k), k


def _ball_frame(b, role, flag, weight_mode):
    r = ROLE[role]
    d = b[b.faced] if role == "bat" else b
    w = {"full": d.ball_weight, "comp_only": d.comp_weight, "none": pd.Series(1.0, index=d.index)}[weight_mode]
    out = pd.DataFrame({
        "pid": d[r["pid"]], "match_id": d.match_id, "pressure": d[flag], "w": w,
        "rv": d[r["rv"]], "wpa": r["wpa_sign"] * d.wpa_bat,
        "runs": d.runs if role == "bat" else d.bowler_runs,
        "out": d.striker_out if role == "bat" else d.bowler_wicket,
        "dot": d.is_dot if role == "bat" else d.is_legal & (d.bowler_runs == 0), "bnd": d.is_boundary, "legal": d.is_legal, "lev": d.leverage,
    })
    return out


def _agg(f):
    wx = lambda c: (f[c] * f.w)
    g = f.assign(rv_w=wx("rv"), rv2_w=f.rv ** 2 * f.w, wpa_w=wx("wpa"), wpa2_w=f.wpa ** 2 * f.w).groupby("pid")
    return g.agg(balls=("w", "size"), w=("w", "sum"), rv=("rv_w", "sum"), rv2=("rv2_w", "sum"),
                 wpa=("wpa_w", "sum"), wpa2=("wpa2_w", "sum"), runs=("runs", "sum"), outs=("out", "sum"),
                 dots=("dot", "sum"), bnd=("bnd", "sum"), legal=("legal", "sum"),
                 matches=("match_id", "nunique"), avg_lev=("lev", "mean"))


def _consistency(fp, players, n_boot=300, min_balls=3, seed=C.SEED):
    m = fp[fp.pid.isin(players)].assign(rv_w=lambda x: x.rv * x.w).groupby(["pid", "match_id"]).agg(
        rv=("rv_w", "sum"), w=("w", "sum"), n=("w", "size")).reset_index()
    m = m[m.n >= min_balls]
    rng = np.random.default_rng(seed)
    rows = {}
    for pid, g in m.groupby("pid"):
        rv, w = g.rv.to_numpy(), g.w.to_numpy()
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        boot = rv[idx].sum(1) / w[idx].sum(1)
        rows[pid] = {"pressure_apps": len(g), "pos_share": (rv > 0).mean(),
                     "boot_p10": np.percentile(boot, 10), "boot_p90": np.percentile(boot, 90)}
    return pd.DataFrame.from_dict(rows, orient="index")


def _z(s):
    return (s - s.mean()) / s.std(ddof=0)


def player_table(b, role, flag="pressure_li", min_balls=None, min_matches=C.MIN_PRESSURE_MATCHES,
                 weight_mode="full", weights=None, seed=C.SEED):
    weights = weights or C.COMPOSITE_WEIGHTS
    min_balls = min_balls or (C.MIN_PRESSURE_BALLS_BAT if role == "bat" else C.MIN_PRESSURE_BALLS_BOWL)
    f = _ball_frame(b, role, flag, weight_mode)
    P, N = _agg(f[f.pressure]), _agg(f[~f.pressure])
    t = P.add_prefix("p_").join(N.add_prefix("n_"), how="outer").fillna(0)

    for pre in ("p_", "n_"):
        ok = t[pre + "w"] > 0
        for met in ("rv", "wpa"):
            sh, _ = eb_shrink(t.loc[ok, pre + met], t.loc[ok, pre + "w"], t.loc[ok, pre + met + "2"])
            t.loc[ok, pre + met + "_eb"] = sh
    t["delta_rv"] = t.p_rv_eb - t.n_rv_eb

    elig = t[(t.p_balls >= min_balls) & (t.p_matches >= min_matches)].copy()
    cons = _consistency(f[f.pressure], elig.index, seed=seed)
    elig = elig.join(cons)

    elig["c_impact"] = _z(elig.p_wpa_eb)
    elig["c_skill"] = _z(elig.p_rv_eb)
    elig["c_delta"] = _z(elig.delta_rv)
    elig["c_consistency"] = 0.5 * _z(elig.pos_share) + 0.5 * _z(elig.boot_p10)
    elig["score"] = sum(weights[k] * elig["c_" + k] for k in weights)
    elig["rank"] = elig.score.rank(ascending=False, method="first").astype(int)
    return elig.sort_values("rank"), t


def describe(tab, b, role, teams):
    """Add readable columns: name, latest team, raw rates."""
    r = ROLE[role]
    last = b.sort_values("match_id").groupby(r["pid"]).agg(
        name=(r["name"], "last"), team_id=(r["team"], "last"), comp_id=("comp_id", "last"),
        ipl_share=("is_ipl", "mean"), first_year=("year", "min"), last_year=("year", "max"))
    tnames = teams.drop_duplicates(["comp_id", "team_id"]).set_index(["comp_id", "team_id"]).team_name
    last["latest_team"] = [tnames.get((c, t), "") for c, t in zip(last.comp_id, last.team_id)]
    out = tab.join(last[["name", "latest_team", "ipl_share", "first_year", "last_year"]])
    out["p_rv_raw"] = out.p_rv / out.p_w
    out["n_rv_raw"] = out.n_rv / out.n_w.replace(0, np.nan)
    if role == "bat":
        out["p_sr"] = 100 * out.p_runs / out.p_balls
        out["n_sr"] = 100 * out.n_runs / out.n_balls.replace(0, np.nan)
        out["p_avg"] = out.p_runs / out.p_outs.replace(0, np.nan)
        out["p_dot_pct"] = 100 * out.p_dots / out.p_balls
        out["p_bnd_pct"] = 100 * out.p_bnd / out.p_balls
    else:
        out["p_econ"] = 6 * out.p_runs / out.p_legal
        out["n_econ"] = 6 * out.n_runs / out.n_legal.replace(0, np.nan)
        out["p_sr_bowl"] = out.p_legal / out.p_outs.replace(0, np.nan)
        out["p_dot_pct"] = 100 * out.p_dots / out.p_balls
        out["p_bnd_pct"] = 100 * out.p_bnd / out.p_balls
    return out
