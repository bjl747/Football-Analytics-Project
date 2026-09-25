"""Engine v2: Kalman ratings + context -> projected margin and total.

Built from the research in docs/RESEARCH_LOG.md:
1. team_game_table: per team-game stats (features.py) + points + market-implied points
2. Kalman filter per statistic (kalman.py), hyperparameters fitted on 2012-2022
3. context: QB starter edge, injury burden, rest/bye, time zones, divisional, playoffs
4. Ridge models for margin and total, trained on every completed game
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_games, load_injuries, load_pbp, load_snaps
from .features import canon, team_game_stats
from .injuries import injury_burden
from .kalman import KalmanParams, run_filter
from .players import qb_game_table, qb_values_asof, replacement_level, team_qb_baseline

PARAMS_PATH = Path(__file__).with_name("kalman_params_nfl.json")
WEIGHTS = {"points": None, "mkt_points": None, "epa": "plays", "epa_luck": "plays", "epa_noto": "plays",
           "success": "plays", "explosive": "plays", "pass_epa": "plays", "rush_epa": "plays",
           "early_epa": "plays", "cpoe": "plays", "proe": None, "sack_rate": "plays", "pass_rate": "plays",
           "st_epa": None, "fg_oe_pts": None, "turnovers": None, "fumbles_lost": None, "ints": None,
           "all_plays": None, "drives": None, "rz_td": "rz_trips"}
PT = {"ARI": -7, "LA": -8, "LAC": -8, "SF": -8, "SEA": -8, "LV": -8, "DEN": -7}  # UTC offsets (non-Eastern)
INJ_GROUPS = ["DB", "DL", "LB", "OL", "SKILL"]

MARGIN_RATINGS = ["mkt_points", "points"]
MARGIN_CONTEXT = ["home", "div", "rest_diff", "home_bye", "away_bye", "tz_diff", "west_early", "qb_d", "playoff"] \
    + [f"d_inj_{c}" for c in INJ_GROUPS]
TOTAL_RATINGS = ["points", "mkt_points", "epa", "pass_epa", "rush_epa", "success", "all_plays", "drives",
                 "proe", "pass_rate", "explosive", "st_epa", "turnovers"]
TOTAL_CONTEXT = ["dome", "prime", "div", "playoff", "season"]


def _games(seasons) -> pd.DataFrame:
    g = load_games()
    g = g[g["season"].isin(seasons)].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(canon)
    return g


def team_game_table(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["mkt_home"] = (g["total_line"] + g["spread_line"]) / 2
    g["mkt_away"] = (g["total_line"] - g["spread_line"]) / 2
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        d = g[["game_id", "season", "week", f"{side}_team", f"{opp}_team", f"{side}_score", f"mkt_{side}", "location"]].copy()
        d.columns = ["game_id", "season", "week", "team", "opp", "points", "mkt_points", "location"]
        d["is_home"] = float(side == "home") * (d["location"] != "Neutral")
        rows.append(d.drop(columns="location"))
    base = pd.concat(rows)
    st = stats.drop(columns=["season", "week", "home_team", "is_home", "game_date", "opp"], errors="ignore")
    return base.merge(st, on=["game_id", "team"], how="left")


def kalman_predictions(tgt: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or json.loads(PARAMS_PATH.read_text())
    out = tgt[["game_id", "team"]].copy()
    for m, w in WEIGHTS.items():
        r = run_filter(tgt, m, KalmanParams(**params[m]), w)
        out = out.merge(r[["game_id", "team", "pred"]].rename(columns={"pred": f"k_{m}"}), on=["game_id", "team"], how="left")
    return out


def qb_edges(games: pd.DataFrame, qbg: pd.DataFrame) -> pd.DataFrame:
    repl = replacement_level(qbg)
    rows = []
    for (s, w), gw in games.groupby(["season", "week"]):
        vals = qb_values_asof(qbg, s, w, repl)
        base = team_qb_baseline(qbg, vals, s, w, repl)
        for r in gw.itertuples():
            e = {}
            for side, team, qb in (("h", r.home_team, r.home_qb_id), ("a", r.away_team, r.away_qb_id)):
                e[side] = float(vals.get(qb, repl) - base.get(team, vals.get(qb, repl))) if isinstance(qb, str) else 0.0
            rows.append({"game_id": r.game_id, "qb_edge_h": e["h"], "qb_edge_a": e["a"]})
    return pd.DataFrame(rows)


def game_frame(games, kp, qbe, inj) -> pd.DataFrame:
    g = games.copy()
    kc = [c for c in kp.columns if c.startswith("k_")]
    h = kp.rename(columns={c: "h_" + c for c in kc}).rename(columns={"team": "home_team"})
    a = kp.rename(columns={c: "a_" + c for c in kc}).rename(columns={"team": "away_team"})
    g = g.merge(h, on=["game_id", "home_team"], how="left").merge(a, on=["game_id", "away_team"], how="left")
    g = g.merge(qbe, on="game_id", how="left")
    for side, col in (("h", "home_team"), ("a", "away_team")):
        cols = {f"inj_{c}": f"{side}_inj_{c}" for c in INJ_GROUPS}
        x = inj.rename(columns={"team": col, **cols})
        g = g.merge(x[["season", "week", col] + [c for c in cols.values() if c in x]], on=["season", "week", col], how="left")
    for c in INJ_GROUPS:
        g[f"d_inj_{c}"] = g.get(f"h_inj_{c}", 0).fillna(0) - g.get(f"a_inj_{c}", 0).fillna(0)
    g["neutral"] = (g["location"] == "Neutral").astype(float)
    g["home"] = 1 - g["neutral"]
    g["rest_diff"] = (g["home_rest"].fillna(7) - g["away_rest"].fillna(7)).clip(-7, 7)
    g["home_bye"] = (g["home_rest"] >= 13).astype(float)
    g["away_bye"] = (g["away_rest"] >= 13).astype(float)
    g["dome"] = g["roof"].isin(["dome", "closed"]).astype(float)
    g["div"] = g["div_game"].fillna(0)
    g["kick_hr"] = g["gametime"].fillna("13:00").str[:2].astype(int)
    g["prime"] = (g["kick_hr"] >= 19).astype(float)
    g["playoff"] = (g["game_type"] != "REG").astype(float)
    g["tz_diff"] = g["away_team"].map(PT).fillna(-5) - g["home_team"].map(PT).fillna(-5)
    g["west_early"] = ((g["away_team"].map(PT).fillna(-5) <= -7) & (g["kick_hr"] <= 13)).astype(float)
    g["qb_d"] = g["qb_edge_h"].fillna(0) - g["qb_edge_a"].fillna(0)
    for m in set(MARGIN_RATINGS) | set(TOTAL_RATINGS):
        g[f"d_{m}"] = g[f"h_k_{m}"] - g[f"a_k_{m}"]
        g[f"s_{m}"] = g[f"h_k_{m}"] + g[f"a_k_{m}"]
    return g


MARGIN_X = [f"d_{m}" for m in MARGIN_RATINGS] + MARGIN_CONTEXT
TOTAL_X = [f"s_{m}" for m in TOTAL_RATINGS] + TOTAL_CONTEXT


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))


class GamePredictor:
    """Weekly NFL game predictor (engine v2)."""

    def __init__(self, first_season: int = 2010):
        self.first = first_season
        self.frame: pd.DataFrame | None = None

    def build(self, last_season: int) -> "GamePredictor":
        seasons = list(range(self.first, last_season + 1))
        pbp = {s: load_pbp(s) for s in seasons}
        stats = pd.concat([team_game_stats(p) for p in pbp.values()])
        qbg = pd.concat([qb_game_table(p) for p in pbp.values()])
        qbg["team"] = qbg["team"].map(canon)
        self.qbg = qbg
        games = _games(seasons)
        kp = kalman_predictions(team_game_table(games, stats))
        inj_seasons = [s for s in seasons if s >= 2012]
        inj = injury_burden(load_injuries(inj_seasons), load_snaps(inj_seasons), canon)
        self.frame = game_frame(games, kp, qb_edges(games, qbg), inj)
        return self

    def fit(self, train_from: int = 2012) -> "GamePredictor":
        f = self.frame
        tr = f[(f["season"] >= train_from) & f["result"].notna()]
        tr_m = tr.dropna(subset=MARGIN_X)
        tr_t = tr.dropna(subset=TOTAL_X)
        self.margin_model = _ridge().fit(tr_m[MARGIN_X], tr_m["result"])
        self.total_model = _ridge().fit(tr_t[TOTAL_X], tr_t["total"])
        return self

    def predict(self, season: int, week: int) -> pd.DataFrame:
        f = self.frame
        wk = f[(f["season"] == season) & (f["week"] == week)].copy()
        wk[MARGIN_X] = wk[MARGIN_X].fillna(0)
        wk["pred_margin"] = self.margin_model.predict(wk[MARGIN_X])
        wk["pred_total"] = self.total_model.predict(wk[TOTAL_X].fillna(wk[TOTAL_X].mean()))
        wk["home_exp"] = (wk["pred_total"] + wk["pred_margin"]) / 2
        wk["away_exp"] = (wk["pred_total"] - wk["pred_margin"]) / 2
        return wk
