"""Quarterback valuation and depth-chart (starter change) adjustments.

The single biggest piece of information a power rating misses is a change at
quarterback (injury, benching, rest). We value every QB by EPA per dropback
using only plays *before* the game being predicted, shrunk toward
replacement level (Bayesian: a QB with few dropbacks is assumed to be close to
a typical backup). The engine then compares the announced starter with the QB
play the team's ratings were built on.

This is also the foundation of the player-value (WAR) work in the secondary
project: the same shrink-and-compare approach extends to other positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def qb_game_table(pbp: pd.DataFrame, epa_col: str = "qb_epa") -> pd.DataFrame:
    """Dropbacks and total EPA per QB per game."""
    d = pbp[(pbp["qb_dropback"] == 1) & pbp["id"].notna() & pbp[epa_col].notna()]
    d = d.assign(cpoe_v=d["cpoe"].fillna(0) / 100.0, cpoe_n=d["cpoe"].notna().astype(float))
    t = d.groupby(["game_id", "season", "week", "posteam", "id"]).agg(
        dropbacks=(epa_col, "size"), epa=(epa_col, "sum"),
        cpoe=("cpoe_v", "sum"), attempts=("cpoe_n", "sum")).reset_index()
    return t.rename(columns={"posteam": "team", "id": "qb_id"})


def replacement_level(qbg: pd.DataFrame, max_career_db: int = 150) -> float:
    """Average EPA/dropback of little-used QBs (backups, spot starters)."""
    tot = qbg.groupby("qb_id")[["dropbacks", "epa"]].sum()
    low = tot[tot["dropbacks"] < max_career_db]
    return float(low["epa"].sum() / max(low["dropbacks"].sum(), 1))


def qb_values_asof(qbg: pd.DataFrame, season: int, week: int, repl: float,
                   prior_db: float = 250.0, season_decay: float = 0.6,
                   cpoe_weight: float = 0.0, cpoe_scale: float = 1.0) -> pd.Series:
    """Shrunk EPA/dropback for every QB using games strictly before (season, week).

    With ``cpoe_weight`` > 0 the value blends in shrunk CPOE (completion % over
    expected, the stickier accuracy stat), converted to EPA units by ``cpoe_scale``.
    """
    past = qbg[(qbg["season"] < season) | ((qbg["season"] == season) & (qbg["week"] < week))]
    w = season_decay ** (season - past["season"]).clip(lower=0)
    db = (past["dropbacks"] * w).groupby(past["qb_id"]).sum()
    epa = (past["epa"] * w).groupby(past["qb_id"]).sum()
    val = (epa + prior_db * repl) / (db + prior_db)
    if cpoe_weight and "cpoe" in past:
        att = (past["attempts"] * w).groupby(past["qb_id"]).sum()
        cp = (past["cpoe"] * w).groupby(past["qb_id"]).sum() / (att + prior_db)
        val = (1 - cpoe_weight) * val + cpoe_weight * (repl + cpoe_scale * cp.reindex(val.index).fillna(0))
    return val


def team_qb_baseline(qbg: pd.DataFrame, values: pd.Series, season: int, week: int,
                     repl: float) -> pd.Series:
    """Dropback-weighted value of the QBs a team has used this season so far
    (last season if no games yet): what the team's ratings already 'contain'."""
    cur = qbg[(qbg["season"] == season) & (qbg["week"] < week)]
    base = cur if len(cur) else qbg[qbg["season"] == season - 1]
    if base.empty:
        return pd.Series(dtype=float)
    v = base["qb_id"].map(values).fillna(repl)
    return (v * base["dropbacks"]).groupby(base["team"]).sum() / base.groupby("team")["dropbacks"].sum()


def qb_edges(games_week: pd.DataFrame, qbg: pd.DataFrame, season: int, week: int,
             repl: float, overrides: dict[str, str] | None = None) -> dict[tuple[str, str], float]:
    """EPA/dropback difference between each team's starter and its baseline.

    ``overrides`` maps team -> QB id to model a lineup change (e.g. injury news).
    Returns {(game_id, team): edge}.
    """
    vals = qb_values_asof(qbg, season, week, repl)
    base = team_qb_baseline(qbg, vals, season, week, repl)
    out = {}
    for _, r in games_week.iterrows():
        for team, qb in ((r["home_team"], r.get("home_qb_id")), (r["away_team"], r.get("away_qb_id"))):
            if overrides and team in overrides:
                qb = overrides[team]
            if not isinstance(qb, str):
                out[(r["game_id"], team)] = 0.0
                continue
            out[(r["game_id"], team)] = float(vals.get(qb, repl) - base.get(team, vals.get(qb, repl)))
    return out
