"""Download and parse free historical NFL OPENING lines (2010-2021) from the
sportsbookreviewsonline.com archive and match them to nflverse game ids.

Why: closing lines are the sharpest numbers in betting. Real bettors act on
lines earlier in the week. A model's true edge must be measured against
the line you could actually have bet, and opening lines are the free proxy.
"""
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import DATA_DIR, load_games  # noqa: E402
from superalgo.features import canon  # noqa: E402

URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{a}-{b:02d}/"
NAMES = {
    "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF", "Carolina": "CAR",
    "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "GreenBay": "GB", "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX",
    "KansasCity": "KC", "KCChiefs": "KC", "Kansas": "KC", "LasVegas": "LV", "Oakland": "LV", "LVRaiders": "LV",
    "LAChargers": "LAC", "SanDiego": "LAC", "LARams": "LA", "St.Louis": "LA", "StLouis": "LA", "LosAngeles": "LA",
    "Miami": "MIA", "Minnesota": "MIN", "NewEngland": "NE", "NewOrleans": "NO", "NYGiants": "NYG",
    "NYJets": "NYJ", "Philadelphia": "PHI", "Pittsburgh": "PIT", "SanFrancisco": "SF", "Seattle": "SEA",
    "TampaBay": "TB", "Tampa": "TB", "Tennessee": "TEN", "Washington": "WAS", "Washingtom": "WAS",
}


def _num(x):
    if isinstance(x, str) and x.strip().lower() in ("pk", "p", "pick"):
        return 0.0
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def parse_season(a: int) -> pd.DataFrame:
    req = urllib.request.Request(URL.format(a=a, b=(a + 1) % 100), headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    t = pd.read_html(StringIO(html))[0]
    t.columns = t.iloc[0]
    t = t.iloc[1:].reset_index(drop=True)
    rows = []
    for i in range(0, len(t) - 1, 2):
        v, h = t.iloc[i], t.iloc[i + 1]
        if v["VH"] not in ("V", "N") or h["VH"] not in ("H", "N"):
            continue
        rec = {"season": a, "date": str(v["Date"]).zfill(4), "away": NAMES.get(v["Team"], v["Team"]),
               "home": NAMES.get(h["Team"], h["Team"])}
        for col in ("Open", "Close"):
            vo, ho = _num(v[col]), _num(h[col])
            if np.isnan(vo) or np.isnan(ho):
                rec[f"spread_{col.lower()}"] = rec[f"total_{col.lower()}"] = np.nan
                continue
            # the favourite's row carries the spread, the other row the total.
            # stored as expected HOME margin (nflverse convention: + = home favoured)
            if vo < ho:   # away favoured by vo
                rec[f"spread_{col.lower()}"], rec[f"total_{col.lower()}"] = -vo, ho
            else:         # home favoured by ho
                rec[f"spread_{col.lower()}"], rec[f"total_{col.lower()}"] = ho, vo
        rows.append(rec)
    d = pd.DataFrame(rows)
    return d


def main():
    frames = []
    for a in range(2010, 2022):
        try:
            frames.append(parse_season(a)); print(a, len(frames[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            print(a, "failed", e)
    d = pd.concat(frames)
    g = load_games()
    g = g[g["season"].between(2010, 2021)].copy()
    g["home"], g["away"] = g["home_team"].map(canon), g["away_team"].map(canon)
    g["date"] = pd.to_datetime(g["gameday"]).dt.strftime("%m%d")
    m = g.merge(d, on=["season", "date", "home", "away"], how="left")
    # neutral / London games may list sides the other way round: try swapped
    miss = m["spread_open"].isna()
    sw = d.rename(columns={"home": "away", "away": "home"})
    for c in ("spread_open", "spread_close"):
        sw[c] = -sw[c]
    m2 = g[miss.values].merge(sw, on=["season", "date", "home", "away"], how="left")
    m.loc[miss.values, ["spread_open", "total_open", "spread_close", "total_close"]] = \
        m2[["spread_open", "total_open", "spread_close", "total_close"]].values
    out = m[["game_id", "spread_open", "total_open", "spread_close", "total_close", "spread_line", "total_line"]]
    print("matched", out["spread_open"].notna().mean(),
          "close agreement", (out["spread_close"] - out["spread_line"]).abs().median())
    out.to_parquet(DATA_DIR / "opening_lines.parquet")


if __name__ == "__main__":
    main()
