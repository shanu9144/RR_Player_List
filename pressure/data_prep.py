import numpy as np
import pandas as pd

from . import config as C


def load_raw():
    d = C.DATA_DIR
    return {
        "balls": pd.read_csv(d / "ball_by_ball.csv", low_memory=False),
        "matches": pd.read_csv(d / "matches.csv"),
        "comps": pd.read_csv(d / "competitions.csv"),
        "teams": pd.read_csv(d / "teams.csv"),
        "xi": pd.read_csv(d / "playing_eleven.csv"),
        "ctp": pd.read_csv(d / "competition_team_player_mapping.csv"),
    }


def _competition_weight(row):
    name = row["comp_name"]
    if name.startswith("IPL"):
        return C.W_IPL
    if row["match_format"] == "T20":
        return C.W_A_TEAMS
    full = (row["home_team"] in C.FULL_MEMBERS) + (row["away_team"] in C.FULL_MEMBERS)
    return {2: C.W_FULL_V_FULL, 1: C.W_FULL_V_ASSOC, 0: C.W_ASSOC_V_ASSOC}[full]


def prepare_matches(raw):
    m = raw["matches"].merge(raw["comps"], on="comp_id", how="left")
    m["match_date"] = pd.to_datetime(m["match_date"])
    m["year"] = m["match_date"].dt.year
    m["is_ipl"] = m["comp_name"].str.startswith("IPL")
    m["comp_weight"] = m.apply(_competition_weight, axis=1)
    age = (C.LATEST_YEAR - m["year"]).clip(lower=0)
    m["recency_weight"] = 0.5 ** (age / C.RECENCY_HALF_LIFE_YEARS)
    m["has_winner"] = m["win_team_id"].notna()
    return m


def prepare_balls(raw, matches):
    b = raw["balls"]
    b = b[b["innings"] <= 2].copy()  # innings 3/4 are super overs
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    et = b["extras_type"].fillna("None")
    b["is_wide"] = et.isin(C.WIDE_TYPES)
    b["is_noball"] = et.isin(C.NO_BALL_TYPES)
    b["is_legal"] = ~(b["is_wide"] | b["is_noball"])
    b["faced"] = ~b["is_wide"]
    for col in ["is_four", "is_six", "is_wicket"]:
        b[col] = b[col].astype(str).str.lower().eq("true")

    b["total_runs"] = b["runs"] + b["extras1_runs"] + b["extras_2_runs"]
    b["bowler_runs"] = (
        b["runs"]
        + np.where(b["is_wide"], b["extras1_runs"] + b["extras_2_runs"], 0)
        + np.where(b["is_noball"], b["extras1_runs"], 0)
    )
    b["bowler_wicket"] = b["is_wicket"] & b["dismissal_type"].isin(C.BOWLER_WICKETS)
    b["team_wicket"] = b["is_wicket"]
    b["striker_out"] = b["is_wicket"] & (b["dismissed_player_id"] == b["striker_id"]) & ~b["dismissal_type"].eq("Retire Out")
    b["is_dot"] = b["faced"] & (b["runs"] == 0) & b["is_legal"]
    b["is_boundary"] = b["is_four"] | b["is_six"]

    g = b.groupby(["match_id", "innings"], sort=False)
    b["score_before"] = g["total_runs"].cumsum() - b["total_runs"]
    b["wkts_before"] = g["team_wicket"].cumsum() - b["team_wicket"].astype(int)
    b["legal_before"] = g["is_legal"].cumsum() - b["is_legal"].astype(int)
    b["innings_total"] = g["total_runs"].transform("sum")

    # wickets in previous 18 legal balls and consecutive dots faced, both pre-ball
    def _rolling(df):
        w = df["team_wicket"].astype(int).rolling(18, min_periods=1).sum().shift(1).fillna(0)
        dots = df["is_dot"].astype(int)
        streak = dots.groupby((dots == 0).cumsum()).cumsum().shift(1).fillna(0)
        return pd.DataFrame({"recent_wkts": w, "dot_streak": streak}, index=df.index)

    roll = g[["team_wicket", "is_dot"]].apply(_rolling).reset_index(level=[0, 1], drop=True)
    b[["recent_wkts", "dot_streak"]] = roll.loc[b.index]

    mcols = ["match_id", "comp_id", "comp_name", "match_format", "year", "is_ipl", "knock_out",
             "win_team_id", "has_winner", "comp_weight", "recency_weight",
             "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]
    b = b.merge(matches[mcols], on="match_id", how="left")

    # revised overs are stored in balls
    max1 = b["first_inn_revised_overs"].fillna(C.FULL_BALLS)
    max2 = b["second_inn_revised_overs"].fillna(C.FULL_BALLS)
    b["max_balls"] = np.where(b["innings"] == 1, max1, max2)
    b["balls_rem"] = (b["max_balls"] - b["legal_before"]).clip(lower=1)

    first_tot = b[b["innings"] == 1].groupby("match_id")["innings_total"].first()
    b["first_inn_total"] = b["match_id"].map(first_tot)
    b["target"] = np.where(b["revised_target"].notna(), b["revised_target"], b["first_inn_total"] + 1)
    b.loc[b["innings"] == 1, "target"] = np.nan
    b["runs_needed"] = b["target"] - b["score_before"]
    b["wkts_in_hand"] = 10 - b["wkts_before"]
    overs_done = b["legal_before"] / 6
    b["crr"] = np.where(overs_done > 0, b["score_before"] / overs_done.replace(0, np.nan), 0.0)
    b["rrr"] = np.where(b["innings"] == 2, b["runs_needed"] * 6 / b["balls_rem"], np.nan)
    b["over_num"] = b["overid"]
    b["phase"] = pd.cut(b["overid"], [0, 6, 15, 20], labels=["powerplay", "middle", "death"]).astype(str)
    b["is_spin"] = b["pace_or_spin"].eq("Spin")
    b["is_lhb"] = b["batting_type"].eq("LHB")
    b["bat_won"] = (b["win_team_id"] == b["batting_team_id"]).astype(float)
    b.loc[~b["has_winner"], "bat_won"] = np.nan
    b["ball_weight"] = b["comp_weight"] * b["recency_weight"]
    return b


def validate(balls, matches):
    """Reconcile rebuilt scores with recorded winners; returns (report_text, mismatch_df)."""
    tot = balls.groupby(["match_id", "innings"]).agg(
        total=("total_runs", "sum"), team=("batting_team_id", "first"),
        wkts=("team_wicket", "sum")).reset_index()
    p = tot.pivot(index="match_id", columns="innings")
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.dropna(subset=["total_1", "total_2"]).join(
        matches.set_index("match_id")[["win_team_id", "revised_target", "result"]])
    p = p[p["win_team_id"].notna()]
    target = p["revised_target"].fillna(p["total_1"] + 1)
    chase_won = p["total_2"] >= target
    tie = p["total_2"] == target - 1
    exp_winner = np.where(chase_won, p["team_2"], p["team_1"])
    ok = (exp_winner == p["win_team_id"]) | tie
    mism = p[~ok]
    lines = [
        f"Balls (innings 1-2): {len(balls):,}",
        f"Matches with balls: {balls['match_id'].nunique()}",
        f"Matches with both innings and a recorded winner: {len(p)}",
        f"Winner reconstructed correctly (ties excluded): {ok.mean():.2%}",
        f"Tied (super-over decided): {int(tie.sum())}",
        f"Matches without winner (no result/abandoned/missing): {int((~matches['has_winner']).sum())}",
        f"Legal balls per completed over == 6: "
        f"{(balls[balls.is_legal].groupby(['match_id','innings','overid']).size() == 6).mean():.2%}",
    ]
    return "\n".join(lines), mism
