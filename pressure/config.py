from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data Pack"
OUT_DIR = ROOT / "outputs"

SEED = 42  # convention, only makes bootstrap and model fits repeatable; other bootstrap seeds keep 9-10 of the top 10
FULL_BALLS = 120  # 20 overs x 6; no innings in the data has more legal balls
LATEST_YEAR = 2026  # newest season in the data (last match 2026-11-01); anchors the recency weights

# Inferred from the rows; this coding gives the most overs with exactly 6 legal balls (98.2% vs 95.9% if only wides are excluded)
NO_BALL_TYPES = {"Overstep", "other"}
WIDE_TYPES = {"wide"}
# Cricket rule: run-outs, retirements and obstruction are not credited to the bowler
BOWLER_WICKETS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}

# The 12 ICC full members in 2022-26 (not in the data); every other team is treated as an associate
FULL_MEMBERS = {
    "India", "Australia", "England", "South Africa", "New Zealand", "Pakistan",
    "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan", "Ireland", "Zimbabwe",
}

# Competition-strength weights (ball-level): judgement, not fitted; halving the weaker-nation ones keeps rank agreement at 0.96+
W_IPL = 1.0
W_FULL_V_FULL = 1.0
W_FULL_V_ASSOC = 0.6
W_ASSOC_V_ASSOC = 0.4
W_A_TEAMS = 0.6

# Judgement: 2026 = 1.0, 2024 = 0.5, 2022 = 0.25; half-lives of 1-5 years keep 8-10 of the top 10, no recency keeps only 5 bowlers
RECENCY_HALF_LIFE_YEARS = 2.0

# Primary settings; the sensitivity grid varies these
# Top 20% (pressure score above 1.44x average): judgement and the most sensitive setting; top 15% or 25% keeps only 4-7 of the top 10
PRESSURE_PCTL = 0.80
# Each pressure ball carries ~2.8 runs of noise, so under 100 balls is mostly luck; 100 keeps 119 batters and 123 bowlers (150 keeps 79 and 86)
MIN_PRESSURE_BALLS_BAT = 100
MIN_PRESSURE_BALLS_BOWL = 100
# Needed for the match-by-match consistency measure; excludes nobody once 100 balls are required
MIN_PRESSURE_MATCHES = 8
# Judgement: impact and skill overlap (correlation ~0.75) so together get 50%; equal 25% each gives the same top 10
COMPOSITE_WEIGHTS = {"impact": 0.30, "skill": 0.20, "delta": 0.25, "consistency": 0.25}
