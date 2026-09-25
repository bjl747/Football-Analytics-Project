"""Experiment 10: QB value = EPA only vs EPA + CPOE composite (brief item)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import run  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from superalgo.data import DATA_DIR  # noqa: E402
from superalgo.players import qb_values_asof, replacement_level, team_qb_baseline  # noqa: E402

g = pd.read_parquet(DATA_DIR / "exp5_frame.parquet")
q = pd.read_parquet(DATA_DIR / "qb_games_all.parquet")
repl = replacement_level(q)
for m in ("mkt_points", "points"):
    g[f"d_{m}"] = g[f"h_k_{m}"] - g[f"a_k_{m}"]
base_ctx = ["home", "div", "rest_diff", "home_bye", "away_bye", "tz_diff", "west_early", "playoff"] + \
    [c for c in g.columns if c.startswith("d_inj_")]
R = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))  # noqa: E731

for wgt in (0.0, 0.3, 0.5):
    d = {}
    for (s, w), gw in g.groupby(["season", "week"]):
        vals = qb_values_asof(q, s, w, repl, cpoe_weight=wgt, cpoe_scale=2.0)
        base = team_qb_baseline(q, vals, s, w, repl)
        for r in gw.itertuples():
            e = []
            for team, qb in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
                e.append(float(vals.get(qb, repl) - base.get(team, vals.get(qb, repl))) if isinstance(qb, str) else 0.0)
            d[r.game_id] = e[0] - e[1]
    g["qb_dx"] = g["game_id"].map(d).fillna(0)
    X = g[["d_mkt_points", "d_points", "qb_dx"] + base_ctx].fillna(0)
    run(g, X, R, label=f"dev  cpoe_weight={wgt}")
    run(g, X, R, holdout=True, label=f"HOLD cpoe_weight={wgt}")
