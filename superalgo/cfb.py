"""College football data from CollegeFootballData.com (CFBD), free tier.

Free key: https://collegefootballdata.com/key. Then:

    export CFBD_API_KEY=your_key_here

This adapter converts CFBD data into the same tables the NFL engine uses, so
the whole ratings -> simulation -> advice pipeline runs for college unchanged:

* ``load_games``     -> games table (scores, neutral site, closing lines)
* ``load_efficiency``-> team-game offensive EPA/play (CFBD calls EPA "PPA")
* ``preseason_prior``-> Bayesian prior from returning production + recruiting,
                        as the research brief specifies

NOT YET TESTED against the live API (no key during the first build). The
simulator also needs a college calibration (more drives, bigger home field
edge), built from CFBD drive data via ``calibrate_cfb``.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://api.collegefootballdata.com"


def _get(path: str, **params) -> list[dict]:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError("Set CFBD_API_KEY (free at https://collegefootballdata.com/key)")
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(f"{BASE}{path}?{q}", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _pick(d: dict, *names):
    """CFBD has used both camelCase and snake_case field names across versions."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


def load_games(year: int, division: str = "fbs") -> pd.DataFrame:
    games = _get("/games", year=year, division=division, seasonType="both")
    lines = {int(_pick(l, "id")): l for l in _get("/lines", year=year)}
    rows = []
    for g in games:
        gid = int(_pick(g, "id"))
        hs, as_ = _pick(g, "homePoints", "home_points"), _pick(g, "awayPoints", "away_points")
        rec = {"game_id": str(gid), "season": year, "week": _pick(g, "week"),
               "game_type": "REG" if _pick(g, "seasonType", "season_type") == "regular" else "POST",
               "gameday": str(_pick(g, "startDate", "start_date"))[:10],
               "home_team": _pick(g, "homeTeam", "home_team"), "away_team": _pick(g, "awayTeam", "away_team"),
               "home_score": hs, "away_score": as_,
               "result": (hs - as_) if hs is not None and as_ is not None else np.nan,
               "total": (hs + as_) if hs is not None and as_ is not None else np.nan,
               "location": "Neutral" if _pick(g, "neutralSite", "neutral_site") else "Home",
               "spread_line": np.nan, "total_line": np.nan}
        ln = lines.get(gid)
        if ln and ln.get("lines"):
            books = ln["lines"]
            pref = next((b for b in books if b.get("provider") in ("consensus", "Bovada", "DraftKings")), books[0])
            sp, tot = _pick(pref, "spread"), _pick(pref, "overUnder", "over_under")
            # CFBD spread is from the home team's view in betting notation (-7 = home favoured)
            rec["spread_line"] = -float(sp) if sp is not None else np.nan
            rec["total_line"] = float(tot) if tot is not None else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def load_efficiency(year: int) -> pd.DataFrame:
    """Team-game offensive EPA/play (garbage time excluded by CFBD)."""
    rows = []
    for g in _get("/ppa/games", year=year, excludeGarbageTime="true"):
        off = _pick(g, "offense")
        overall = off.get("overall") if isinstance(off, dict) else None
        if overall is None:
            continue
        rows.append({"game_id": str(_pick(g, "gameId", "game_id")), "posteam": _pick(g, "team"),
                     "defteam": _pick(g, "opponent"), "epa_play": float(overall),
                     "plays": 65.0, "season": year, "week": _pick(g, "week")})
    eff = pd.DataFrame(rows)
    if eff.empty:
        return eff
    games = load_games(year).set_index("game_id")
    eff["home_team"] = eff["game_id"].map(games["home_team"])
    eff["date"] = eff["game_id"].map(games["gameday"])
    eff["is_home"] = (eff["posteam"] == eff["home_team"]).astype(float)
    return eff


def preseason_prior(year: int, w_returning: float = 8.0, w_recruiting: float = 6.0) -> dict[str, float]:
    """Per-team net point adjustments for the Bayesian prior.

    Returning production (share of last season's offensive/defensive PPA that
    comes back) and 4-year recruiting talent, each standardised, then scaled to
    points. The weights are starting guesses to be fitted by backtest once data
    is available.
    """
    ret = pd.DataFrame(_get("/player/returning", year=year))
    rec = pd.DataFrame(_get("/recruiting/teams", year=year))
    adj = pd.Series(0.0, index=pd.Index(sorted(set(ret.get("team", [])) | set(rec.get("team", [])))))
    if len(ret):
        col = next(c for c in ("percentPPA", "percent_ppa", "totalPPA") if c in ret)
        z = ret.set_index("team")[col].astype(float)
        adj = adj.add(w_returning * (z - z.mean()) / z.std(), fill_value=0)
    if len(rec):
        z = rec.set_index("team")["points"].astype(float)
        adj = adj.add(w_recruiting * (z - z.mean()) / z.std(), fill_value=0)
    return adj.to_dict()


def calibrate_cfb(year: int) -> dict:
    """Drive-level constants for the simulator from CFBD drive data."""
    dr = pd.DataFrame(_get("/drives", year=year))
    res = dr[_col(dr, "driveResult", "drive_result")].str.upper()
    per_game = dr.groupby(_col(dr, "gameId", "game_id")).size()
    td = res.str.contains("TD") & ~res.str.contains("INT TD|FUMBLE TD|PUNT TD", regex=True)
    fg = res.eq("FG")
    pts_per_drive = 6.96 * td.mean() + 3 * fg.mean()
    return {"drives_per_game_mean": float(per_game.mean()), "drives_per_game_sd": float(per_game.std()),
            "base_rates": {"td": float(td.mean()), "fg": float(fg.mean()),
                           "opp_td": float(res.str.contains("INT TD|FUMBLE TD", regex=True).mean()),
                           "safety": float(res.eq("SF").mean())},
            "points_per_drive": float(pts_per_drive)}


def _col(df, *names):
    return next(n for n in names if n in df)
