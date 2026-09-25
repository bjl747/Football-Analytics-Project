"""Experiment 5: best pure prediction model (no current-week line used).

Adds context to the Kalman ratings: QB starter edge, injury burden, rest/bye,
time zones and travel, divisional games. Compares Ridge vs gradient boosting.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import load_game_frame, margin_features, run  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from superalgo.data import DATA_DIR  # noqa: E402

g = load_game_frame()
e2 = pd.read_parquet(DATA_DIR / "exp2_frame.parquet")[["game_id", "qb_edge_h", "qb_edge_a"]]
g = g.merge(e2, on="game_id", how="left")
inj = pd.read_parquet(DATA_DIR / "injury_burden.parquet")
ic = [c for c in inj.columns if c.startswith("inj_") and c != "inj_QB"]
for side, col in (("h", "home_team"), ("a", "away_team")):
    x = inj.rename(columns={"team": col, **{c: f"{side}_{c}" for c in ic}})[["season", "week", col] + [f"{side}_{c}" for c in ic]]
    g = g.merge(x, on=["season", "week", col], how="left")
for c in ic:
    g[f"d_{c}"] = g[f"h_{c}"].fillna(0) - g[f"a_{c}"].fillna(0)
PT = {"ARI": -7, "LA": -8, "LAC": -8, "SF": -8, "SEA": -8, "LV": -8, "DEN": -7}
g["tz_diff"] = g["away_team"].map(PT).fillna(-5) - g["home_team"].map(PT).fillna(-5)
g["kick_hr"] = g["gametime"].fillna("13:00").str[:2].astype(int)
g["west_early"] = ((g["away_team"].map(PT).fillna(-5) <= -7) & (g["kick_hr"] <= 13)).astype(float)
g["home_bye"] = (g["home_rest"] >= 13).astype(float)
g["away_bye"] = (g["away_rest"] >= 13).astype(float)
g["qb_d"] = g["qb_edge_h"].fillna(0) - g["qb_edge_a"].fillna(0)
g.to_parquet(DATA_DIR / "exp5_frame.parquet")

RIDGE = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))  # noqa: E731
XGB = lambda: XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.02, subsample=0.8,  # noqa: E731
                           colsample_bytree=0.8, min_child_weight=20, reg_lambda=10)
PERF = ["points", "epa", "epa_luck", "pass_epa", "rush_epa", "early_epa", "success", "cpoe", "st_epa",
        "turnovers", "fumbles_lost", "all_plays", "drives", "sack_rate", "explosive"]
CTX = ["home", "div", "rest_diff", "home_bye", "away_bye", "tz_diff", "west_early", "qb_d", "playoff"]
INJ = [f"d_{c}" for c in ic]


def feats(ms, extra):
    X = margin_features(g, ms)
    for c in extra:
        X[c] = g[c].fillna(0)
    return X


for label, X in (("perf+mkt", feats(PERF + ["mkt_points"], ["home"])),
                 ("perf+mkt+ctx", feats(PERF + ["mkt_points"], CTX)),
                 ("perf+mkt+ctx+inj", feats(PERF + ["mkt_points"], CTX + INJ)),
                 ("mkt+ctx+inj", feats(["mkt_points", "points"], CTX + INJ))):
    run(g, X, RIDGE, label="ridge " + label)
    run(g, X, XGB, label="xgb   " + label)
