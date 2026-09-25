"""In-game Win Probability as a Generalized Additive Model (GAM).

Each input gets its own smooth spline curve (so the model can learn, e.g.,
that a 7-point lead matters far more with 2 minutes left than with 50), and
the curves are added together inside a logistic link. Built from
scikit-learn's SplineTransformer + LogisticRegression, which is a penalised
logistic GAM without needing any extra (paid or niche) libraries.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

FEATURES = [
    "score_differential", "game_seconds_remaining", "diff_time_ratio",
    "spread_time", "yardline_100", "down", "ydstogo",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "is_home",
]


def add_wp_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    secs = df["game_seconds_remaining"].clip(lower=0)
    df["is_home"] = (df["posteam"] == df["home_team"]).astype(float)
    # point spread from the offence's view (nflverse spread_line = expected home margin)
    spread = df.get("spread_line", pd.Series(0.0, index=df.index)).fillna(0.0)
    pos_spread = np.where(df["is_home"] == 1, spread, -spread)
    # the pre-game line matters less as the clock runs down
    df["spread_time"] = pos_spread * np.exp(-4.0 * (1 - secs / 3600.0))
    # a lead is worth more the less time is left
    df["diff_time_ratio"] = df["score_differential"] / np.sqrt(secs + 60.0) * 10
    df["down"] = df["down"].fillna(1)
    df["ydstogo"] = df["ydstogo"].fillna(10)
    df["yardline_100"] = df["yardline_100"].fillna(75)
    for c in ("posteam_timeouts_remaining", "defteam_timeouts_remaining"):
        df[c] = df[c].fillna(3)
    return df


def _usable(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp[pbp["posteam"].notna() & pbp["score_differential"].notna()
             & pbp["game_seconds_remaining"].notna()]
    return add_wp_features(df)


class WinProbModel:
    def __init__(self, n_knots: int = 8, C: float = 1.0):
        self.model = make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear"),
            LogisticRegression(C=C, max_iter=2000),
        )

    def fit(self, pbp: pd.DataFrame) -> "WinProbModel":
        df = _usable(pbp)
        home_margin = df["result"]  # nflverse: home score - away score
        df = df[home_margin != 0]
        won = np.where(df["is_home"] == 1, df["result"] > 0, df["result"] < 0)
        self.model.fit(df[FEATURES].astype(float), won.astype(int))
        return self

    def predict(self, states: pd.DataFrame) -> np.ndarray:
        """Probability the team with the ball wins."""
        df = add_wp_features(states)
        return self.model.predict_proba(df[FEATURES].astype(float))[:, 1]

    def add_wp(self, pbp: pd.DataFrame) -> pd.DataFrame:
        df = _usable(pbp)
        df["my_wp"] = self.model.predict_proba(df[FEATURES].astype(float))[:, 1]
        return df
