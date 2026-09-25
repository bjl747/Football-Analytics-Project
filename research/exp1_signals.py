"""Experiment 1: which Kalman-rated statistics predict game margins?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import load_game_frame, margin_features, run  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

g = load_game_frame()
M = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))  # noqa: E731
ALL = ["points", "epa", "epa_luck", "epa_noto", "success", "explosive", "pass_epa", "rush_epa",
       "early_epa", "cpoe", "proe", "sack_rate", "pass_rate", "st_epa", "fg_oe_pts", "turnovers",
       "fumbles_lost", "ints", "all_plays", "drives", "rz_td"]
sets = {
    "points only": ["points"],
    "market ratings only": ["mkt_points"],
    "epa only": ["epa"],
    "epa_luck only": ["epa_luck"],
    "epa_noto only": ["epa_noto"],
    "success only": ["success"],
    "pass+rush epa": ["pass_epa", "rush_epa"],
    "points+epa": ["points", "epa"],
    "all performance stats": ALL,
    "all + market": ALL + ["mkt_points"],
}
for name, ms in sets.items():
    X = margin_features(g, ms)
    X["home"] = g["home"]
    run(g, X, M, label=name)
