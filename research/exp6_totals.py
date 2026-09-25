"""Experiment 6: game totals from Kalman ratings (dev CV + holdout)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import run, total_features  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from superalgo.data import DATA_DIR  # noqa: E402

g = pd.read_parquet(DATA_DIR / "exp5_frame.parquet")
R = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))  # noqa: E731
SETS = {"points": ["points"], "mkt": ["mkt_points"],
        "perf+mkt": ["points", "mkt_points", "epa", "pass_epa", "rush_epa", "success", "all_plays", "drives",
                     "proe", "pass_rate", "explosive", "st_epa", "turnovers"]}
for name, ms in SETS.items():
    X = total_features(g, ms)
    for c in ("dome", "prime", "div", "playoff"):
        X[c] = g[c]
    X["season"] = g["season"]  # scoring environment drifts by era
    for ho in (False, True):
        run(g, X, R, target="total", line="total_line", holdout=ho, label=("HOLD " if ho else "dev  ") + name)
