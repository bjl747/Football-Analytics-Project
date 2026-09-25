"""Live sportsbook odds (The Odds API, free tier) and market consensus.

Free tier: 500 requests/month at https://the-odds-api.com. One request
returns every NFL game x every US book x spreads/totals/moneylines, so a
few calls per day fit comfortably. Set the key in the environment:

    export ODDS_API_KEY=your_key_here

The realistic free edge is *line shopping*: our fair probability (model
blended with the sharp-book consensus) versus the best price across many
books. Soft books often hang stale or off-market numbers.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .odds import devig

API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORTS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf"}
# books whose prices best reflect the "true" market; weighted more in consensus
SHARP_BOOKS = {"pinnacle": 3.0, "circasports": 2.0, "betonlineag": 1.5, "lowvig": 1.5}

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def fetch_odds(league: str = "nfl", api_key: str | None = None, regions: str = "us,us2,eu",
               markets: str = "h2h,spreads,totals") -> list[dict]:
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Set ODDS_API_KEY (free at https://the-odds-api.com)")
    q = urllib.parse.urlencode({"apiKey": key, "regions": regions, "markets": markets,
                                "oddsFormat": "american"})
    with urllib.request.urlopen(f"{API.format(sport=SPORTS[league])}?{q}", timeout=30) as r:
        return json.loads(r.read())


def odds_to_frame(events: list[dict], league: str = "nfl") -> pd.DataFrame:
    """Flatten The Odds API response: one row per book x market x outcome."""
    name = (lambda t: NFL_TEAMS.get(t, t)) if league == "nfl" else (lambda t: t)
    rows = []
    for ev in events:
        home, away = name(ev["home_team"]), name(ev["away_team"])
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for oc in mk.get("outcomes", []):
                    sel = oc["name"]
                    rows.append({"event_id": ev["id"], "commence": ev["commence_time"],
                                 "home_team": home, "away_team": away, "book": bk["key"],
                                 "market": mk["key"],
                                 "side": name(sel) if sel not in ("Over", "Under") else sel,
                                 "point": oc.get("point"), "price": oc["price"]})
    return pd.DataFrame(rows)


def consensus(odds: pd.DataFrame) -> pd.DataFrame:
    """Market consensus per game: home spread (expected home margin), total,
    and no-vig home win probability, weighted toward sharp books."""
    out = []
    for (eid, home, away), g in odds.groupby(["event_id", "home_team", "away_team"]):
        rec = {"event_id": eid, "home_team": home, "away_team": away}
        wts = g["book"].map(SHARP_BOOKS).fillna(1.0)
        sp = g[(g["market"] == "spreads") & (g["side"] == home)]
        if len(sp):
            rec["mkt_spread"] = float(-np.average(sp["point"], weights=wts[sp.index]))
        tot = g[(g["market"] == "totals") & (g["side"] == "Over")]
        if len(tot):
            rec["mkt_total"] = float(np.average(tot["point"], weights=wts[tot.index]))
        ml = g[g["market"] == "h2h"].pivot_table(index="book", columns="side", values="price")
        if home in ml and away in ml:
            ml = ml.dropna(subset=[home, away])
            probs = [devig(h, a)[0] for h, a in zip(ml[home], ml[away])]
            if probs:
                rec["mkt_p_home"] = float(np.average(probs, weights=ml.index.map(lambda b: SHARP_BOOKS.get(b, 1.0))))
        out.append(rec)
    return pd.DataFrame(out)


def offers(odds: pd.DataFrame, event_id: str) -> list[dict]:
    """Every (book, market, side, point, price) offer for one game."""
    return odds[odds["event_id"] == event_id].to_dict("records")
