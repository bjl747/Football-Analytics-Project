"""Measure the league-wide constants the simulator needs, straight from
play-by-play data, and save them to JSON so apps never need to recompute.

What we learn:
* how many drives a game has (pace) and its spread;
* how a team's points-per-drive splits into TD vs FG rates;
* rare drive outcomes (defensive TD, safety);
* dual-state "garbage time" behaviour: how late-game drive outcomes change
  when a team is leading or trailing by different amounts;
* how often a regulation tie ends tied after overtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BUCKETS = [-100, -17, -9, -1, 0, 8, 16, 100]
BUCKET_LABELS = ["trail17+", "trail9-16", "trail1-8", "tied", "lead1-8", "lead9-16", "lead17+"]
OUTCOMES = ["td", "fg", "opp_td", "safety"]
LATE_SECONDS = 360  # final 6 minutes of regulation

DEFAULT_PATH = Path(__file__).with_name("calibration_nfl.json")


def drive_table(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna() & (pbp["qtr"] <= 4)]
    dr = p.groupby(["game_id", "fixed_drive"]).agg(
        team=("posteam", "first"), res=("fixed_drive_result", "first"),
        start_sec=("game_seconds_remaining", "max"), diff=("score_differential", "first"),
    ).reset_index()
    dr["td"] = dr["res"].eq("Touchdown")
    dr["fg"] = dr["res"].eq("Field goal")
    dr["opp_td"] = dr["res"].eq("Opp touchdown")
    dr["safety"] = dr["res"].eq("Safety")
    dr["late"] = dr["start_sec"] <= LATE_SECONDS
    dr["bucket"] = pd.cut(dr["diff"], BUCKETS, labels=BUCKET_LABELS)
    return dr


def calibrate(pbp: pd.DataFrame, games: pd.DataFrame | None = None) -> dict:
    pbp = pbp[pbp["season_type"] == "REG"] if "season_type" in pbp else pbp
    dr = drive_table(pbp)
    per_game = dr.groupby("game_id").size()

    # points-per-drive -> TD / FG rate split, measured at team-game level
    early = dr[~dr["late"]]
    tg = early.groupby(["game_id", "team"]).agg(n=("td", "size"), td=("td", "mean"), fg=("fg", "mean"))
    tg = tg[tg["n"] >= 6]
    ppd = 6.96 * tg["td"] + 3 * tg["fg"]
    # TD share of scoring rises with offensive quality
    td_slope, td_int = np.polyfit(ppd, tg["td"], 1)

    base = early[OUTCOMES].mean()
    late = dr[dr["late"]].groupby("bucket", observed=True)[OUTCOMES].mean()
    normal = early.groupby("bucket", observed=True)[OUTCOMES].mean()
    # late-game multipliers relative to that same score state earlier in the game
    mult = (late + 0.005) / (normal + 0.005)

    # extra points: share of TDs followed by a made PAT / 2-pt try
    xp = pbp[pbp["extra_point_result"].notna()]
    two = pbp[pbp["two_point_conv_result"].notna()]
    n_try = len(xp) + len(two)
    cal = {
        "drives_per_game_mean": float(per_game.mean()),
        "drives_per_game_sd": float(per_game.std()),
        "late_drive_share": float(dr["late"].mean()),
        "td_rate_slope": float(td_slope),
        "td_rate_intercept": float(td_int),
        "base_rates": {k: float(v) for k, v in base.items()},
        "late_multipliers": {b: {k: float(mult.loc[b, k]) for k in OUTCOMES} for b in mult.index},
        "pat": {
            "kick_share": len(xp) / n_try,
            "kick_make": float(xp["extra_point_result"].eq("good").mean()),
            "two_make": float(two["two_point_conv_result"].eq("success").mean()) if len(two) else 0.48,
        },
        "form_sd": 0.25,
        "pace_sd_scale": 1.0,
    }
    if games is not None:
        g = games.dropna(subset=["result", "spread_line", "total_line"])
        g = g[g["game_type"] == "REG"] if "game_type" in g else g
        cal["target_margin_resid_sd"] = float((g["result"] - g["spread_line"]).std())
        cal["target_total_resid_sd"] = float((g["total"] - g["total_line"]).std())
        cal["league_avg_points"] = float(pd.concat([g["home_score"], g["away_score"]]).mean())
        loc = g["location"].eq("Home") if "location" in g else True
        cal["hfa"] = float(g.loc[loc, "result"].mean())
    return cal


def save(cal: dict, path: Path = DEFAULT_PATH) -> None:
    Path(path).write_text(json.dumps(cal, indent=2))


def load(path: Path = DEFAULT_PATH) -> dict:
    return json.loads(Path(path).read_text())
