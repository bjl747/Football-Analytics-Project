"""Experiment 8: is the model worth more vs the market early in the season?"""
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
for m in ("mkt_points", "points"):
    g[f"d_{m}"] = g[f"h_k_{m}"] - g[f"a_k_{m}"]
ctx = ["home", "div", "rest_diff", "home_bye", "away_bye", "tz_diff", "west_early", "qb_d", "playoff"] + \
      [c for c in g.columns if c.startswith("d_inj_")]
X = g[["d_mkt_points", "d_points"] + ctx].fillna(0)
R = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))  # noqa: E731
p1, i1 = cv_predict(g, X, g["result"], R)
p2, i2 = cv_predict(g, X, g["result"], R, holdout=True)
d = g.loc[np.concatenate([i1, i2])].assign(pred=np.concatenate([p1, p2]))


def w(df):
    dd, r = df.pred - df.spread_line, df.result - df.spread_line
    return round(float(np.clip(np.dot(dd, r) / np.dot(dd, dd), 0, 1)), 3), len(df)


for lab, per in (("dev", d.season <= 2022), ("hold", d.season >= 2023)):
    x = d[per]
    print(lab, "weeks 1-4", w(x[x.week <= 4]), "weeks 5-9", w(x[x.week.between(5, 9)]),
          "weeks 10+", w(x[x.week >= 10]))
