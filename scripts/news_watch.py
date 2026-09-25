"""Check ESPN injury news for this week's games and flag QB changes.

    python scripts/news_watch.py --season 2026 --week 3
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import load_games, load_pbp  # noqa: E402
from superalgo.features import canon  # noqa: E402
from superalgo.news import fetch_injuries, qb_alerts  # noqa: E402
from superalgo.players import qb_game_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, required=True)
ap.add_argument("--week", type=int, required=True)
a = ap.parse_args()

g = load_games()
wk = g[(g.season == a.season) & (g.week == a.week)].copy()
for c in ("home_team", "away_team"):
    wk[c] = wk[c].map(canon)
qbg = pd.concat([qb_game_table(load_pbp(s)) for s in range(a.season - 3, a.season + 1)])
qbg["team"] = qbg["team"].map(canon)
inj = fetch_injuries()
alerts = qb_alerts(wk, inj, qbg, a.season, a.week)
print(json.dumps(alerts, indent=2) if alerts else "No starting-QB injury alerts right now.")
print(f"\n{len(inj)} injury entries; Out/Doubtful by team:")
print(inj[inj.status.isin(["Out", "Doubtful"])].groupby("team").size().sort_values(ascending=False).head(10).to_string())
