"""Experiment 9: can we predict which way the line moves (open -> close)?

Useful even without an ATS edge: it tells a bettor whether to bet now or wait.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import cv_predict  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from superalgo.data import DATA_DIR  # noqa: E402

g = pd.read_parquet(DATA_DIR / "exp5_frame.parquet")
op = pd.read_parquet(DATA_DIR / "opening_lines.parquet")[["game_id", "spread_open"]]
es = pd.read_parquet(DATA_DIR / "espn_lines.parquet")[["game_id", "spread_open"]]
g = g.drop(columns=[c for c in g.columns if c == "spread_open"]).merge(
    pd.concat([op, es]).dropna().drop_duplicates("game_id"), on="game_id", how="left")
for m in ("mkt_points", "points"):
    g[f"d_{m}"] = g[f"h_k_{m}"] - g[f"a_k_{m}"]
import os
ctx = ["home", "div", "rest_diff", "home_bye", "away_bye", "tz_diff", "west_early", "playoff"] + (["qb_d"] if os.environ.get("WITH_WEEK_NEWS") else []) + \
      ([c for c in g.columns if c.startswith("d_inj_")] if os.environ.get("WITH_WEEK_NEWS") else [])
X = g[["d_mkt_points", "d_points"] + ctx].fillna(0)
R = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))  # noqa: E731
p1, i1 = cv_predict(g, X, g["result"], R)
p2, i2 = cv_predict(g, X, g["result"], R, holdout=True)
d = g.loc[np.concatenate([i1, i2])].assign(pred=np.concatenate([p1, p2])).dropna(subset=["spread_open"])
d["dis"] = d["pred"] - d["spread_open"]
d["move"] = d["spread_line"] - d["spread_open"]
for lab, per in (("dev 2012-21", d.season <= 2022), ("hold 2024-25", d.season >= 2024)):
    x = d[per & (d.move != 0)]
    for k in (0, 1.5, 3):
        y = x[x.dis.abs() >= k]
        hit = (np.sign(y.dis) == np.sign(y.move)).mean()
        gain = (np.sign(y.dis) * y.move).mean()  # avg points of line value gained by betting at open
        print(f"{lab} |model-open|>={k}: move-direction hit {hit:.3f}  avg line value gained {gain:+.2f} pts  n={len(y)}")
