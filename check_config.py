"""Evidence behind the values in pressure/config.py: python check_config.py

Needs outputs/cache_scored_balls.pkl (made by run_pipeline.py). Changes one setting at a time and reports how
many of the main top 10 survive. Writes outputs/09_config_*.
"""
import warnings

import pandas as pd
from scipy.stats import kendalltau

from pressure import config as C
from pressure.data_prep import load_raw
from pressure.metrics import player_table
from pressure.pressure_index import add_pressure

warnings.filterwarnings("ignore")
OUT = C.OUT_DIR
PARTS = ["c_impact", "c_skill", "c_delta", "c_consistency"]


def data_checks(raw, b):
    r = raw["balls"][raw["balls"].innings <= 2]
    et = r.extras_type.fillna("None")
    legal = r[~et.isin(C.WIDE_TYPES | C.NO_BALL_TYPES)].groupby(["match_id", "innings"]).size()
    lines = [f"Latest match date: {pd.to_datetime(raw['matches'].match_date).max().date()}",
             f"Most legal balls in any innings: {legal.max()} (innings with exactly {C.FULL_BALLS}: {(legal == C.FULL_BALLS).mean():.1%})",
             "Overs with exactly 6 legal balls, by how extras are coded (higher = better):"]
    for label, bad in [("wide + Overstep + other (used)", {"wide", "Overstep", "other"}),
                       ("wide + Overstep", {"wide", "Overstep"}), ("wide + other", {"wide", "other"}), ("wide only", {"wide"})]:
        share = (r[~et.isin(bad)].groupby(["match_id", "innings", "overid"]).size() == 6).mean()
        lines.append(f"  {label:32s} {share:.2%}")
    lines.append(f"Dismissal types: {r.dismissal_type.value_counts().to_dict()}")
    w = b.groupby("comp_weight").size() / len(b)
    lines.append("Share of balls by competition weight: " + ", ".join(f"{k}: {v:.1%}" for k, v in w.items()))
    q = b.leverage.quantile([0.5, 0.75, 0.8, 0.85, 0.9, 0.95])
    lines.append("Pressure score at percentiles: " + ", ".join(f"{k:.0%}: {v:.2f}" for k, v in q.items()))
    lines.append(f"Spread (st.dev) of run value per pressure ball: batters {b[b.faced & b.pressure_li].rv_bat.std():.2f}, "
                 f"bowlers {b[b.pressure_li].rv_bowl.std():.2f}")
    return "\n".join(lines)


def eligibility(b):
    rows = []
    for role in ("bat", "bowl"):
        _, t = player_table(b, role)
        for mb in (50, 75, 100, 150, 200):
            rows.append({"role": role, "min_balls": mb,
                         **{f"min_matches={mm}": int(((t.p_balls >= mb) & (t.p_matches >= mm)).sum()) for mm in (5, 8, 10)}})
    return pd.DataFrame(rows)


def compare(base, alt):
    common = base.index.intersection(alt.index)
    return {"top10_kept": len(set(base.index[:10]) & set(alt.index[:10])),
            "rank_agreement_tau": round(kendalltau(base.loc[common, "rank"], alt.loc[common, "rank"]).statistic, 2),
            "n_eligible": len(alt)}


def one_at_a_time(b, role, base):
    age = (C.LATEST_YEAR - b.year).clip(lower=0)
    rows = {}

    def run(label, data=b, **kw):
        rows[label] = compare(base, player_table(data, role, **kw)[0])

    for p in (0.70, 0.75, 0.85, 0.90):
        run(f"pressure = top {round((1 - p) * 100)}%", b.assign(flag=b.li_pct >= p), flag="flag")
    run("rule-based pressure (top 20%)", b.assign(flag=b.rule_pct >= C.PRESSURE_PCTL), flag="flag")
    run("no recency weight", weight_mode="comp_only")
    run("no competition or recency weights", weight_mode="none")
    for h in (1, 3, 5):
        run(f"recency half-life {h} yr", b.assign(ball_weight=b.comp_weight * 0.5 ** (age / h)))
    halved = b.comp_weight.where(b.comp_weight >= 1.0, b.comp_weight / 2)
    run("weaker-nation weights halved", b.assign(ball_weight=halved * b.recency_weight))
    run("all competitions equal (recency kept)", b.assign(ball_weight=b.recency_weight))
    for label, w in {"equal 25% each": {k: 0.25 for k in C.COMPOSITE_WEIGHTS},
                     "no consistency part": {"impact": 0.4, "skill": 0.3, "delta": 0.3, "consistency": 0.0},
                     "impact only": {"impact": 1.0, "skill": 0.0, "delta": 0.0, "consistency": 0.0}}.items():
        run(f"rating weights: {label}", weights=w)
    for mb in (50, 150, 200):
        run(f"min balls = {mb}", min_balls=mb)
    for mm in (5, 10, 12):
        run(f"min matches = {mm}", min_matches=mm)
    for s in (1, 7, 123):
        run(f"bootstrap seed = {s}", seed=s)
    return pd.DataFrame(rows).T.rename_axis("change")


def main():
    b = add_pressure(pd.read_pickle(OUT / "cache_scored_balls.pkl"), C.PRESSURE_PCTL)
    text = [data_checks(load_raw(), b)]
    for role in ("bat", "bowl"):
        base, _ = player_table(b, role)
        text.append(f"\n{role}: {len(base)} eligible, median pressure balls {base.p_balls.median():.0f}; "
                    f"correlation between the four rating parts:\n{base[PARTS].corr().round(2).to_string()}")
        res = one_at_a_time(b, role, base)
        res.to_csv(OUT / f"09_config_checks_{role}.csv")
        text.append(f"\nChange ONE setting - {role} (base = main settings):\n{res.to_string()}")
    elig = eligibility(b)
    elig.to_csv(OUT / "09_config_eligibility.csv", index=False)
    text.append(f"\nEligible players by minimum balls and matches:\n{elig.to_string(index=False)}")
    (OUT / "09_config_data_checks.txt").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
