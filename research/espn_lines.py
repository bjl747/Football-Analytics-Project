"""Opening + closing lines for recent NFL seasons from ESPN's free (unofficial)
odds feed (sportsbook: ESPN BET / DraftKings), matched to nflverse game ids.

Fills the 2022+ gap left by the sportsbookreviewsonline archive so the
holdout seasons can be tested against opening lines too.
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import DATA_DIR, load_games  # noqa: E402
from superalgo.market import NFL_TEAMS  # noqa: E402

SB = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={y}&seasontype={t}&week={w}"
ODDS = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{e}/competitions/{e}/odds"


def _get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            time.sleep(2 ** i)
    return {}


def _f(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return np.nan


def events(season):
    out = []
    for t, weeks in ((2, range(1, 19)), (3, range(1, 6))):
        for w in weeks:
            d = _get(SB.format(y=season, t=t, w=w))
            for e in d.get("events", []):
                comp = e["competitions"][0]
                teams = {c["homeAway"]: c["team"]["displayName"] for c in comp["competitors"]}
                out.append({"event": e["id"], "season": season, "date": e["date"][:10],
                            "home": NFL_TEAMS.get(teams.get("home")), "away": NFL_TEAMS.get(teams.get("away"))})
    return out


def odds(ev):
    d = _get(ODDS.format(e=ev["event"]))
    items = [i for i in d.get("items", []) if "Live" not in i.get("provider", {}).get("name", "")]
    if not items:
        return ev
    it = items[0]
    h = it.get("homeTeamOdds", {})
    ev["book"] = it.get("provider", {}).get("name")
    ev["close_home_spread"] = _f(it.get("spread"))
    ev["close_total"] = _f(it.get("overUnder"))
    ev["open_home_spread"] = _f(h.get("open", {}).get("pointSpread", {}).get("american"))
    tot = it.get("open", {}).get("total", {})
    ev["open_total"] = _f(tot.get("alternateDisplayValue", "").lstrip("ou")) if tot else np.nan
    return ev


def main(seasons=(2022, 2023, 2024, 2025)):
    evs = [e for s in seasons for e in events(s)]
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(odds, evs))
    d = pd.DataFrame(rows)
    g = load_games()
    g = g[g["season"].isin(seasons)].copy()
    m = []
    for r in d.itertuples():
        c = g[(g.season == r.season) & (g.home_team == r.home) & (g.away_team == r.away)
              & (abs(pd.to_datetime(g.gameday) - pd.to_datetime(r.date)).dt.days <= 1)]
        if len(c):
            m.append({"game_id": c.game_id.iloc[0], **r._asdict()})
    m = pd.DataFrame(m)
    # nflverse convention: expected home margin (+ = home favoured)
    m["spread_open"] = -m["open_home_spread"]
    m["spread_close_espn"] = -m["close_home_spread"]
    m = m.rename(columns={"open_total": "total_open", "close_total": "total_close_espn"})
    m.to_parquet(DATA_DIR / "espn_lines.parquet")
    print(len(d), "events", len(m), "matched", m[["spread_open", "total_open"]].notna().mean().to_dict())


if __name__ == "__main__":
    main()
