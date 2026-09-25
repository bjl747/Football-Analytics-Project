"""Expected Points (EP) and Expected Points Added (EPA).

An XGBoost multi-class model predicts which of seven "next score" events
follows the current game state (from the offence's point of view):

    0 Touchdown (+7)   1 Opp Touchdown (-7)
    2 Field Goal (+3)  3 Opp Field Goal (-3)
    4 Safety (+2)      5 Opp Safety (-2)      6 No score in half (0)

EP is the probability-weighted value of those outcomes. EPA for a play is
EP after the play minus EP before it (or points scored minus EP before, on a
scoring play).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

CLASSES = ["TD", "Opp_TD", "FG", "Opp_FG", "Safety", "Opp_Safety", "No_Score"]
CLASS_POINTS = np.array([7.0, -7.0, 3.0, -3.0, 2.0, -2.0, 0.0])

FEATURES = [
    "down", "ydstogo", "yardline_100", "half_seconds_remaining",
    "goal_to_go", "is_home", "is_dome", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "era",
]


def _scoring_team(df: pd.DataFrame) -> pd.Series:
    """Which team scored on each play (None if nobody)."""
    team = pd.Series(None, index=df.index, dtype=object)
    td = df["touchdown"].fillna(0).eq(1) & df["td_team"].notna()
    team[td] = df.loc[td, "td_team"]
    fg = df["field_goal_result"].eq("made")
    team[fg] = df.loc[fg, "posteam"]
    sf = df["safety"].fillna(0).eq(1)
    team[sf] = df.loc[sf, "defteam"]
    return team


def _score_type(df: pd.DataFrame) -> pd.Series:
    kind = pd.Series(None, index=df.index, dtype=object)
    kind[df["touchdown"].fillna(0).eq(1) & df["td_team"].notna()] = "TD"
    kind[df["field_goal_result"].eq("made")] = "FG"
    kind[df["safety"].fillna(0).eq(1)] = "Safety"
    return kind


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_home"] = (df["posteam"] == df["home_team"]).astype(float)
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(float)
    df["era"] = df["season"].astype(float)
    df["goal_to_go"] = df["goal_to_go"].fillna(0).astype(float)
    return df


def label_next_score(df: pd.DataFrame) -> pd.Series:
    """Label each play with the index of the next scoring event in the same half."""
    df = df.sort_values(["game_id", "play_id"])
    scorer = _scoring_team(df)
    kind = _score_type(df)
    labels = np.full(len(df), 6, dtype=int)

    half = np.where(df["qtr"] <= 2, 1, 2)  # overtime grouped with 2nd half
    key = df["game_id"].astype(str).values + "_" + half.astype(str)
    posteam = df["posteam"].values
    scorer_v, kind_v = scorer.values, kind.values
    next_scorer, next_kind, last_key = None, None, None
    # walk backwards so each play "sees" the next score after it
    for i in range(len(df) - 1, -1, -1):
        if key[i] != last_key:
            next_scorer, next_kind, last_key = None, None, key[i]
        if isinstance(kind_v[i], str) and isinstance(scorer_v[i], str):
            next_scorer, next_kind = scorer_v[i], kind_v[i]
        if next_scorer is not None and isinstance(posteam[i], str):
            own = next_scorer == posteam[i]
            base = {"TD": 0, "FG": 2, "Safety": 4}[next_kind]
            labels[i] = base if own else base + 1
    return pd.Series(labels, index=df.index).reindex(df.index)


def _model_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Scrimmage plays with a valid game state."""
    df = pbp[pbp["down"].notna() & pbp["posteam"].notna() & pbp["yardline_100"].notna()]
    df = df[df["play_type"].isin(["pass", "run", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"])]
    return add_features(df)


class EPModel:
    """XGBoost next-score classifier producing Expected Points."""

    def __init__(self, **xgb_params):
        params = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                      objective="multi:softprob", num_class=len(CLASSES),
                      eval_metric="mlogloss", tree_method="hist", n_jobs=-1)
        params.update(xgb_params)
        self.model = XGBClassifier(**params)

    def fit(self, pbp: pd.DataFrame) -> "EPModel":
        labels = label_next_score(pbp)
        df = _model_frame(pbp)
        y = labels.loc[df.index]
        # weight plays closer to the next score slightly less noisy; keep simple: uniform
        self.model.fit(df[FEATURES].astype(float), y)
        return self

    def predict_proba(self, states: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(states[FEATURES].astype(float))

    def expected_points(self, states: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(states) @ CLASS_POINTS

    def add_epa(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Return scrimmage plays with ``my_ep`` and ``my_epa`` columns."""
        df = _model_frame(pbp).sort_values(["game_id", "play_id"]).copy()
        df["my_ep"] = self.expected_points(df)

        half = np.where(df["qtr"] <= 2, 1, 2)
        grp = df["game_id"].astype(str) + "_" + pd.Series(half, index=df.index).astype(str)
        next_ep = df.groupby(grp)["my_ep"].shift(-1)
        next_pos = df.groupby(grp)["posteam"].shift(-1)
        sign = np.where(next_pos == df["posteam"], 1.0, -1.0)
        ep_after = np.where(next_ep.isna(), 0.0, next_ep * sign)

        scorer = _scoring_team(df)
        kind = _score_type(df)
        pts = kind.map({"TD": 7.0, "FG": 3.0, "Safety": 2.0})
        scored = scorer.notna()
        own = scorer == df["posteam"]
        ep_after = np.where(scored, np.where(own, pts, -pts), ep_after)
        # penalties that are wiped out ("no_play") still move the chains; kept.
        df["my_epa"] = ep_after - df["my_ep"]
        return df
