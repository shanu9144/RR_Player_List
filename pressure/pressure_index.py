"""Pressure definitions: primary (Leverage Index) and rule-based (robustness check)."""
import numpy as np
import pandas as pd


def rule_based_score(b):
    """Transparent 0-1 pressure score: a 'contest is alive' gate times situational intensity."""
    chase = b.innings == 2
    # chase: required rate near a demanding-but-gettable level, wider when wickets are in hand
    width = 2.0 + 0.35 * b.wkts_in_hand
    gate2 = np.exp(-((b.rrr - 9.0) / width) ** 2)
    # first innings: projected total close to the competition par
    proj = b.score_before + b.crr.clip(lower=6) * b.balls_rem / 6 * (1 - 0.05 * b.wkts_before)
    gate1 = np.exp(-((proj - b.par) / 35) ** 2)
    gate = np.where(chase, gate2, 0.8 * gate1)

    phase = b.phase.map({"powerplay": 0.4, "middle": 0.3, "death": 1.0}).to_numpy()
    wk = np.exp(-((b.wkts_before - 5) / 2.5) ** 2)
    collapse = (b.recent_wkts >= 2).astype(float)
    dots = (b.dot_streak / 4).clip(0, 1)
    intensity = 0.45 * phase + 0.25 * wk + 0.20 * collapse + 0.10 * dots
    ko = b.knock_out.fillna(0).astype(float)
    return pd.Series(gate * (0.4 + 0.6 * intensity) + 0.1 * ko, index=b.index)


def component_correlations(b):
    """Spearman correlation of intuitive pressure signals with the Leverage Index, per innings."""
    sig = pd.DataFrame({
        "wickets_down": b.wkts_before, "death_overs": b.phase.eq("death").astype(float),
        "recent_collapse": (b.recent_wkts >= 2).astype(float), "dot_streak": b.dot_streak,
        "required_rate": b.rrr, "wp_closeness": 1 - 2 * (b.wp_before - 0.5).abs(),
        "rule_score": b.rule_score,
    })
    return pd.DataFrame({f"innings_{i}": sig[b.innings == i].corrwith(b.leverage[b.innings == i], method="spearman")
                         for i in (1, 2)})


def add_pressure(b, pctl):
    b = b.copy()
    b["rule_score"] = rule_based_score(b)
    b["li_pct"] = b.leverage.rank(pct=True)
    b["rule_pct"] = b.rule_score.rank(pct=True)
    b["pressure_li"] = b.li_pct >= pctl
    b["pressure_rule"] = b.rule_pct >= pctl
    b["li_band"] = pd.cut(b.li_pct, [0, 0.5, 0.8, 0.95, 1.0], labels=["low", "normal", "high", "extreme"],
                          include_lowest=True).astype(str)
    return b


def agreement(b):
    return {
        "spearman_li_vs_rule": b[["leverage", "rule_score"]].corr(method="spearman").iloc[0, 1],
        "pressure_overlap_jaccard": (b.pressure_li & b.pressure_rule).sum() / (b.pressure_li | b.pressure_rule).sum(),
        "share_pressure_balls_inn2": b.loc[b.pressure_li, "innings"].eq(2).mean(),
        "share_pressure_balls_death": b.loc[b.pressure_li, "phase"].eq("death").mean(),
    }
