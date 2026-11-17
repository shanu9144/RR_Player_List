"""Separate auction watchlists; performance rankings are never modified."""
import pandas as pd


USER_EXCLUDED_IDS = {47, 6819, 3647, 1699, 6288, 2921}
BLOCKED_TEAMS = {"Pakistan", "Pakistan A"}


def screen_players(ratings, market, squad_mapping, as_of="2026-11-17"):
    required = {"player_id", "auction_status", "source_url", "checked_on", "auction_season"}
    if not required.issubset(market.columns):
        raise ValueError(f"Market evidence is missing columns: {sorted(required - set(market.columns))}")
    if market.player_id.duplicated().any():
        raise ValueError("Market evidence must have one row per player ID")
    result = ratings.copy()
    result.index.name = "player_id"
    result = result.join(market.set_index("player_id"), how="left", validate="one_to_one")
    blocked_ids = set(squad_mapping.loc[squad_mapping.team_name.isin(BLOCKED_TEAMS), "player_id"])
    current_squads = squad_mapping[squad_mapping.comp_name.eq("IPL 2026")]
    contracted_ids = set(current_squads.player_id)
    result["excluded_by_request"] = result.index.isin(USER_EXCLUDED_IDS | blocked_ids)
    result["in_2026_ipl_squad"] = result.index.isin(contracted_ids)
    checked = pd.to_datetime(result.checked_on, errors="coerce")
    dated_evidence = result.source_url.fillna("").str.startswith("https://") & checked.notna() & checked.le(pd.Timestamp(as_of))
    result["unsold_watchlist_eligible"] = (
        result.auction_status.eq("unsold") & dated_evidence
        & ~result.excluded_by_request & ~result.in_2026_ipl_squad
    )
    result["availability_label"] = "Unconfirmed; do not assume available"
    result.loc[result.auction_status.eq("sold"), "availability_label"] = "Sold in past auction; not a current free target"
    result.loc[result.in_2026_ipl_squad, "availability_label"] = "In supplied IPL 2026 squad; release must be verified"
    result.loc[result.unsold_watchlist_eligible, "availability_label"] = "Past unsold watchlist; next registration/release still unconfirmed"
    result.loc[result.excluded_by_request, "availability_label"] = "Excluded by user policy"
    return result
