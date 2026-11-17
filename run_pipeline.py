"""End-to-end pipeline: python run_pipeline.py [--refresh]

Writes all tables, charts and a run log to outputs/.
"""
import argparse
import json
import logging
import sys
import time

import pandas as pd

from pressure import charts, config as C
from pressure.auction import screen_players
from pressure.data_prep import load_raw, prepare_balls, prepare_matches, validate
from pressure.expected import add_expected
from pressure.metrics import describe, player_table
from pressure.pressure_index import add_pressure, agreement, component_correlations
from pressure.sensitivity import one_at_a_time, run_grid
from pressure.squad_fit import pick_profiles, rr_gap_table, rr_squad
from pressure.win_prob import cross_fit, evaluate, temporal_holdout

OUT = C.OUT_DIR
CACHE = OUT / "cache_scored_balls.pkl"
log = logging.getLogger("pipeline")


def step(msg):
    log.info("=" * 10 + " " + msg)


def build_scored_balls(raw, refresh):
    if CACHE.exists() and not refresh:
        log.info("Loading cached scored balls: %s", CACHE)
        return pd.read_pickle(CACHE)

    step("1. Data preparation & validation")
    matches = prepare_matches(raw)
    balls = prepare_balls(raw, matches)
    report, mism = validate(balls, matches)
    log.info("\n%s", report)
    (OUT / "01_data_validation.txt").write_text(report)
    mism.to_csv(OUT / "01_winner_mismatches.csv")

    step("2. Win-probability model (5-fold, grouped by match) -> WPA & Leverage Index")
    balls, outcome_p = cross_fit(balls)
    metrics, calib = evaluate(balls)
    log.info("WP out-of-fold: %s", json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    hold_metrics, hold_calib = temporal_holdout(balls)
    log.info("WP temporal holdout (%s): %s", C.LATEST_YEAR, json.dumps({k: round(v, 4) for k, v in hold_metrics.items()}))
    pd.DataFrame({"out_of_fold": metrics, f"holdout_{C.LATEST_YEAR}": hold_metrics}).to_csv(OUT / "02_wp_model_metrics.csv")
    calib.to_csv(OUT / "02_wp_calibration.csv")
    outcome_p.to_csv(OUT / "02_outcome_probabilities.csv")
    charts.calibration(calib, OUT / "02_wp_calibration.png")

    step("3. Expected runs / wickets per ball (out-of-fold)")
    balls, lam = add_expected(balls)
    log.info("Wicket run value by phase: %s", lam.round(2).to_dict())
    lam.to_csv(OUT / "03_wicket_value_by_phase.csv")

    balls.to_pickle(CACHE)
    return balls


def main(refresh=False):
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "run_log.txt", mode="w"), logging.StreamHandler(sys.stdout)])
    t0 = time.time()
    raw = load_raw()
    charts.data_flow({k: len(v) for k, v in raw.items()}, OUT / "01_data_flow.png")
    balls = build_scored_balls(raw, refresh)
    charts.calibration(pd.read_csv(OUT / "02_wp_calibration.csv"), OUT / "02_wp_calibration.png")

    step("4. Pressure definition")
    balls = add_pressure(balls, C.PRESSURE_PCTL)
    agr = agreement(balls)
    log.info("Pressure agreement: %s", json.dumps({k: round(v, 3) for k, v in agr.items()}))
    pd.Series(agr).to_csv(OUT / "04_pressure_agreement.csv")
    comp = component_correlations(balls)
    log.info("Spearman of signals with leverage:\n%s", comp.round(3).to_string())
    comp.to_csv(OUT / "04_signal_vs_leverage_spearman.csv")
    balls.groupby("li_band").agg(balls=("leverage", "size"), mean_lev=("leverage", "mean"),
                                 inn2_share=("innings", lambda s: (s == 2).mean()),
                                 death_share=("phase", lambda s: (s == "death").mean())
                                 ).to_csv(OUT / "04_leverage_bands.csv")
    charts.leverage_heatmap(balls, OUT / "04_leverage_heatmap.png")
    charts.pressure_by_over(balls, OUT / "04_pressure_by_over.png")
    charts.chase_li_curve(balls, OUT / "04_chase_leverage_curve.png")
    top_moments = balls.nlargest(25, "leverage")[
        ["match_id", "comp_name", "innings", "overid", "over_ball_num", "score_before", "wkts_before",
         "target", "striker_name", "bowler_name", "total_runs", "is_wicket", "wp_before", "wp_after", "leverage"]]
    top_moments.to_csv(OUT / "04_highest_leverage_balls.csv", index=False)

    for role in ("bat", "bowl"):
        step(f"5. Player ratings - {role}")
        tab, full = player_table(balls, role)
        log.info("%s: %d players with pressure balls, %d eligible", role, (full.p_balls > 0).sum(), len(tab))

        step(f"6. Sensitivity grid - {role}")
        summ, R, rec, taus = run_grid(balls, role)
        log.info("%s: %d scenarios, Kendall tau vs base: median %.3f, min %.3f",
                 role, R.shape[1], taus.median(), taus.min())
        R.to_csv(OUT / f"06_{role}_scenario_ranks.csv")
        taus.to_csv(OUT / f"06_{role}_kendall_tau.csv")
        base = f"li|{C.PRESSURE_PCTL}|{C.MIN_PRESSURE_BALLS_BAT if role == 'bat' else C.MIN_PRESSURE_BALLS_BOWL}|full|base"
        oat = one_at_a_time(taus, base)
        oat.to_csv(OUT / f"06_{role}_one_at_a_time.csv", index=False)
        log.info("%s one-at-a-time tau:\n%s", role, oat.groupby("changed").kendall_tau.agg(["median", "min"]).round(3).to_string())

        tab = tab.join(summ[["median_rank", "mean_rank", "top10_rate", "scenarios_eligible", "best_rank", "worst_rank"]])
        # mean rank penalises players who drop out or fall away in some scenarios
        tab = tab.sort_values(["mean_rank", "median_rank"])
        tab["final_rank"] = range(1, len(tab) + 1)
        tab = describe(tab, balls, role, raw["teams"])
        tab.to_csv(OUT / f"05_{role}_ratings_full.csv")

        cols = ["final_rank", "name", "latest_team", "score", "rank", "median_rank", "mean_rank", "top10_rate",
                "p_balls", "p_matches", "pressure_apps", "p_rv_raw", "n_rv_raw", "p_rv_eb", "n_rv_eb", "delta_rv", "p_wpa_eb",
                "pos_share", "boot_p10", "ipl_share", "last_year"]
        cols += (["p_sr", "n_sr", "p_avg", "p_dot_pct", "p_bnd_pct"] if role == "bat"
                 else ["p_econ", "n_econ", "p_sr_bowl", "p_dot_pct", "p_bnd_pct"])
        tab[cols].head(25).to_csv(OUT / f"07_{role}_top25.csv")
        log.info("\n%s", tab[cols[:8]].head(15).to_string())
        charts.pressure_vs_normal(tab, role, OUT / f"07_{role}_pressure_vs_normal.png")
        charts.top10_bars(tab, role, OUT / f"07_{role}_top10.png")

        step(f"8. RR squad fit - {role}")
        squad = set(rr_squad(raw).player_id)
        prof = pick_profiles(balls, tab.head(10).index, role)
        prof = tab.head(10)[["final_rank", "name", "latest_team"]].join(prof)
        prof["in_rr_2026_squad"] = prof.index.isin(squad)
        prof.to_csv(OUT / f"08_{role}_pick_profiles.csv")

        # These market screens are separate from the historical rankings and
        # preserve the original top-25 outputs. Historical unsold status only
        # creates a provisional watchlist, never a claim of current availability.
        market = pd.read_csv(C.ROOT / f"auction_{role}_market_evidence.csv")
        auction = screen_players(tab, market, raw["ctp"])
        auction = auction.join(pick_profiles(balls, auction.index, role), how="left")
        eligible = auction[auction.unsold_watchlist_eligible].sort_values(
            ["final_rank", "top10_rate", "p_rv_eb"], ascending=[True, False, False]
        ).copy()
        eligible["auction_watchlist_rank"] = range(1, len(eligible) + 1)
        auction.to_csv(OUT / f"09_{role}_auction_screened.csv")
        eligible.to_csv(OUT / f"09_{role}_auction_watchlist.csv")
        log.info("Auction-evidence eligible %s: %d", role, len(eligible))

    gap = rr_gap_table(balls, raw["teams"])
    log.info("RR vs league, IPL 2025-26 pressure balls:\n%s", gap.round(3).to_string())
    gap.to_csv(OUT / "08_rr_pressure_gaps.csv", index=False)
    rr_squad(raw).to_csv(OUT / "08_rr_squad_ipl2026.csv", index=False)
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="rebuild WP/expected models instead of using cache")
    main(ap.parse_args().refresh)
