"""Injury-news watcher (the "speed edge" from Experiment 9).

Polls ESPN's free injury feed. When a team's expected starting QB is listed
Out or Doubtful, it estimates how far the point spread *should* move, so a
bettor can act before sportsbooks adjust. Run every few minutes Wed-Sun.
"""
from __future__ import annotations

import json
import urllib.request

import pandas as pd

from .players import qb_values_asof, replacement_level

ESPN_INJ = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_TEAM = {"WSH": "WAS", "LAR": "LA", "JAC": "JAX"}
DROPBACKS_PER_GAME = 36.0  # converts EPA/dropback differences into points


def fetch_injuries() -> pd.DataFrame:
    with urllib.request.urlopen(ESPN_INJ, timeout=30) as r:
        d = json.loads(r.read())
    rows = []
    for team in d.get("injuries", []):
        for i in team.get("injuries", []):
            a = i.get("athlete", {})
            abbr = a.get("team", {}).get("abbreviation")
            rows.append({"team": ESPN_TEAM.get(abbr, abbr), "player": a.get("displayName"),
                         "pos": a.get("position", {}).get("abbreviation"), "status": i.get("status"),
                         "updated": i.get("date"), "headline": i.get("shortComment")})
    return pd.DataFrame(rows)


def qb_alerts(week_games: pd.DataFrame, injuries: pd.DataFrame, qbg: pd.DataFrame,
              season: int, week: int) -> list[dict]:
    """Starting QBs ruled Out/Doubtful, with the estimated spread impact.

    week_games needs home/away team, and home/away QB id and name (nflverse schedule).
    """
    repl = replacement_level(qbg)
    vals = qb_values_asof(qbg, season, week, repl)
    cur = qbg[qbg["season"] == season]
    out = []
    bad = injuries[injuries["pos"].eq("QB") & injuries["status"].isin(["Out", "Doubtful"])]
    for r in week_games.itertuples():
        for team, qb_id, qb_name, sign in ((r.home_team, r.home_qb_id, r.home_qb_name, 1),
                                           (r.away_team, r.away_qb_id, r.away_qb_name, -1)):
            hit = bad[(bad["team"] == team) & (bad["player"] == qb_name)]
            if hit.empty:
                continue
            others = cur[(cur["team"] == team) & (cur["qb_id"] != qb_id)]
            backup = others.groupby("qb_id")["dropbacks"].sum().idxmax() if len(others) else None
            v_start = vals.get(qb_id, repl)
            v_back = vals.get(backup, repl) if backup else repl
            shift = sign * (v_back - v_start) * DROPBACKS_PER_GAME
            out.append({"game": f"{r.away_team} @ {r.home_team}", "team": team, "qb": qb_name,
                        "status": hit.iloc[0]["status"], "updated": hit.iloc[0]["updated"],
                        "home_margin_shift": round(float(shift), 1),
                        "note": hit.iloc[0]["headline"]})
    return out
