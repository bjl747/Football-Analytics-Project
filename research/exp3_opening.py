"""Experiment 3: beat the OPENING line / predict line movement.

Model margins come from out-of-sample predictions (leave-one-season-out on
2012-2022, train-on-2012-2022 for the 2024-2025 holdout). Only information
available before the week starts is used (Kalman ratings from past games).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import DEV, HOLD, cv_predict, load_game_frame, margin_features  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from superalgo.data import DATA_DIR  # noqa: E402

g = load_game_frame()
op = pd.read_parquet(DATA_DIR / "opening_lines.parquet")[["game_id", "spread_open", "total_open"]]
es = pd.read_parquet(DATA_DIR / "espn_lines.parquet")[["game_id", "spread_open", "total_open"]]
lines = pd.concat([op, es]).dropna(subset=["spread_open"]).drop_duplicates("game_id")
g = g.merge(lines, on="game_id", how="left")

M = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))  # noqa: E731
ALL = ["points", "epa", "epa_luck", "pass_epa", "rush_epa", "early_epa", "success", "cpoe",
       "st_epa", "turnovers", "fumbles_lost", "all_plays", "drives", "sack_rate", "explosive"]
SETS = {"points": ["points"], "market-kalman": ["mkt_points"], "perf-all": ALL,
        "perf+market": ALL + ["mkt_points"]}


def oos(X):
    X = X.assign(home=g["home"])
    p1, i1 = cv_predict(g, X, g["result"], M)
    p2, i2 = cv_predict(g, X, g["result"], M, holdout=True)
    return pd.Series(np.concatenate([p1, p2]), index=np.concatenate([i1, i2]))


def ats(d, pred, line, k):
    diff = pred - d[line]
    sel = diff.abs() >= k
    r = d["result"] - d[line]
    ok = sel & (r != 0)
    return round(float((np.sign(r[ok]) == np.sign(diff[ok])).mean()), 3), int(ok.sum())


for name, ms in SETS.items():
    pred = oos(margin_features(g, ms))
    d = g.loc[pred.index].assign(pred=pred)
    d = d[d["spread_open"].notna()]
    dev, hold = d[d.season <= 2022], d[d.season >= 2024]
    mv = d["spread_line"] - d["spread_open"]
    # does model disagreement with the opener predict the direction the line moves?
    dis = d["pred"] - d["spread_open"]
    c = np.corrcoef(dis, mv)[0, 1]
    moved = mv != 0
    agree = (np.sign(dis[moved]) == np.sign(mv[moved])).mean()
    print(f"{name:<14} corr(model-open, move)={c:.3f} move-direction hit={agree:.3f}")
    for k in (1.5, 3, 5):
        print(f"   k={k}: vs OPEN dev {ats(dev, dev.pred, 'spread_open', k)} hold {ats(hold, hold.pred, 'spread_open', k)}"
              f" | vs CLOSE dev {ats(dev, dev.pred, 'spread_line', k)} hold {ats(hold, hold.pred, 'spread_line', k)}")
