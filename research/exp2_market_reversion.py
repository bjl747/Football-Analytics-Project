"""Experiment 2: is the 'market overreaction' signal real?

signal = Kalman-smoothed market rating margin (from PAST closing lines) minus
this game's closing spread. Positive = the smoothed market view likes the home
team more than today's line does.
Checks: per-season stability, holdout, and whether QB changes explain it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import load_game_frame  # noqa: E402
from superalgo.data import DATA_DIR  # noqa: E402
from superalgo.players import qb_values_asof, replacement_level, team_qb_baseline  # noqa: E402

g = load_game_frame()
g = g[g["season"].between(2012, 2025) & g["result"].notna() & g["spread_line"].notna()].copy()
g["kmkt_margin"] = g["h_k_mkt_points"] - g["a_k_mkt_points"]
g["signal"] = g["kmkt_margin"] - g["spread_line"]

# QB change flags: starter differs from team's most-used QB so far this season
q = pd.read_parquet(DATA_DIR / "qb_games_all.parquet")
repl = replacement_level(q)
edges = {}
for (s, w), gw in g.groupby(["season", "week"]):
    vals = qb_values_asof(q, s, w, repl)
    base = team_qb_baseline(q, vals, s, w, repl)
    for r in gw.itertuples():
        for side, team, qb in (("h", r.home_team, r.home_qb_id), ("a", r.away_team, r.away_qb_id)):
            edges[(r.game_id, side)] = float(vals.get(qb, repl) - base.get(team, vals.get(qb, repl))) if isinstance(qb, str) else 0.0
g["qb_edge_h"] = [edges[(i, "h")] for i in g["game_id"]]
g["qb_edge_a"] = [edges[(i, "a")] for i in g["game_id"]]
g["qb_change"] = (g["qb_edge_h"].abs() > 0.03) | (g["qb_edge_a"].abs() > 0.03)
g.to_parquet(DATA_DIR / "exp2_frame.parquet")


def ats(d, k):
    sel = d[d["signal"].abs() >= k]
    r = sel["result"] - sel["spread_line"]
    win = np.sign(r) == np.sign(sel["signal"])
    ok = r != 0
    return win[ok].mean(), int(ok.sum())


for k in (1, 2, 3, 4, 5, 6):
    print(f"|signal|>={k}: all {ats(g, k)}  no-QB-change {ats(g[~g.qb_change], k)}  QB-change {ats(g[g.qb_change], k)}")
print("by season (|signal|>=3):")
for s, d in g.groupby("season"):
    print(s, ats(d, 3))
print("dev 2012-22", ats(g[g.season <= 2022], 3), "holdout 2023-25", ats(g[g.season >= 2023], 3))
print("early weeks (<=4)", ats(g[g.week <= 4], 3), "later", ats(g[g.week > 4], 3))
# regression: result - line ~ signal
d = g
b = np.polyfit(d["signal"], d["result"] - d["spread_line"], 1)
print("slope of ATS residual on signal", b)
