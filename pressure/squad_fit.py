"""Rajasthan Royals fit: where RR lose value under pressure, and each pick's phase/matchup profile."""
import numpy as np
import pandas as pd

RR = "Rajasthan Royals"


def rr_squad(raw, comp_name="IPL 2026"):
    ctp = raw["ctp"]
    return ctp[(ctp.comp_name == comp_name) & (ctp.team_name == RR)][["player_id", "player_name"]]


def rr_gap_table(b, teams, seasons=(2025, 2026)):
    ipl_comps = b.loc[b.is_ipl & b.year.isin(seasons), "comp_id"].unique()
    rr_ids = set(teams[(teams.comp_id.isin(ipl_comps)) & (teams.team_name == RR)].team_id)
    d = b[b.is_ipl & b.year.isin(seasons) & b.pressure_li]
    rows = []
    for side, team_col, rv_col, mask in (("batting", "batting_team_id", "rv_bat", d.faced),
                                         ("bowling", "bowling_team_id", "rv_bowl", np.ones(len(d), bool))):
        x = d[mask].assign(is_rr=lambda t: t[team_col].isin(rr_ids))
        g = x.groupby(["phase", "is_rr"])[rv_col].agg(["mean", "size"]).unstack("is_rr")
        for phase in ("powerplay", "middle", "death"):
            rows.append({"side": side, "phase": phase,
                         "rr_rv_per_ball": g.loc[phase, ("mean", True)], "rr_balls": int(g.loc[phase, ("size", True)]),
                         "league_rv_per_ball": g.loc[phase, ("mean", False)]})
    out = pd.DataFrame(rows)
    out["rr_gap_per_ball"] = out.rr_rv_per_ball - out.league_rv_per_ball
    out["rr_gap_per_over"] = 6 * out.rr_gap_per_ball
    return out


def pick_profiles(b, picks, role):
    """Phase and matchup split of each pick's pressure balls."""
    if role == "bat":
        d = b[b.faced & b.pressure_li & b.striker_id.isin(picks)]
        pid, rv, split = "striker_id", "rv_bat", "is_spin"
    else:
        d = b[b.pressure_li & b.bowler_id.isin(picks)]
        pid, rv, split = "bowler_id", "rv_bowl", "is_lhb"
    phase = d.pivot_table(index=pid, columns="phase", values=rv, aggfunc=["mean", "size"])
    phase.columns = [f"{'rv' if a == 'mean' else 'balls'}_{p}" for a, p in phase.columns]
    sp = d.pivot_table(index=pid, columns=split, values=rv, aggfunc="mean")
    sp.columns = [("vs_spin" if role == "bat" else "vs_lhb") if c else ("vs_pace" if role == "bat" else "vs_rhb")
                  for c in sp.columns]
    sp = sp.add_prefix("rv_")
    prof = phase.join(sp)
    if role == "bat":
        prof["median_bat_pos"] = d.groupby(pid).batting_order.median()
        prof["chase_share"] = d.groupby(pid).innings.apply(lambda s: (s == 2).mean())
    else:
        prof["bowling_type"] = d.groupby(pid).bowling_type.agg(lambda s: s.mode().iat[0])
        prof["pace_or_spin"] = d.groupby(pid).pace_or_spin.agg(lambda s: s.mode().iat[0])
    tot = prof.filter(like="balls_").sum(1)
    for p in ("powerplay", "middle", "death"):
        prof[f"share_{p}"] = prof.get(f"balls_{p}", 0) / tot
    return prof
