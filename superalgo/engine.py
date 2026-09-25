"""The game prediction engine: ratings -> expected points -> simulation.

Pipeline for any game:
1. Bayesian least-squares *points* ratings (offence/defence) from scores.
2. Opponent-adjusted *efficiency* ratings from garbage-time-filtered EPA/play.
3. A small linear "stacker", trained only on past seasons' out-of-sample
   predictions, blends them (plus rest, wind and roof) into each team's
   expected points.
4. The dual-state Monte Carlo simulator turns expected points into full
   probability distributions for spreads, totals, moneylines and exact scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import players as P
from . import ratings as R
from .epa import EPModel
from .simulate import GameSimulator, SimResult
from .win_prob import WinProbModel

STACK_FEATURES = ["pts_pred", "mkt_pred", "epa_pred", "qb_edge", "opp_qb_edge", "is_home", "rest_edge", "wind",
                  "is_dome", "early"]


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["neutral"] = g.get("location", pd.Series("Home", index=g.index)).eq("Neutral")
    g["gameday"] = pd.to_datetime(g["gameday"])
    for c in ("home_rest", "away_rest"):
        g[c] = g[c].fillna(7) if c in g else 7
    g["wind"] = g["wind"].fillna(0) if "wind" in g else 0
    g["is_dome"] = g["roof"].isin(["dome", "closed"]).astype(float) if "roof" in g else 0.0
    # market-implied team scores from past closing lines (public before each game)
    if "spread_line" in g and "total_line" in g:
        g["mkt_home_pts"] = (g["total_line"] + g["spread_line"]) / 2
        g["mkt_away_pts"] = (g["total_line"] - g["spread_line"]) / 2
    return g


def build_efficiency(pbp: pd.DataFrame, ep_model: EPModel, wp_model: WinProbModel,
                     wp_bounds=(0.01, 0.99)) -> pd.DataFrame:
    """Team-game EPA/play from our own EP and WP models, garbage time removed."""
    plays = ep_model.add_epa(pbp)
    plays["my_wp"] = wp_model.predict(plays)
    plays["gameday"] = plays["game_date"]
    eff = R.team_game_efficiency(plays, wp_bounds=wp_bounds)
    eff["season"] = eff["game_id"].str[:4].astype(int)
    return eff


@dataclass
class EngineConfig:
    prior_keep: float = 0.6          # share of last season's rating carried over
    prior_games: float = 4.0         # prior weight in pseudo-games
    epa_prior_plays: float = 600.0   # prior weight in pseudo-plays
    half_life_days: float | None = None  # optional recency weighting inside a season
    ridge_alpha: float = 1.0
    w_margin: float = 0.07           # model weight vs market for spreads (from backtest)
    w_total: float = 0.12            # model weight vs market for totals (from backtest)


class GameEngine:
    def __init__(self, sim: GameSimulator | None = None, config: EngineConfig | None = None,
                 qb_games: pd.DataFrame | None = None):
        self.sim = sim or GameSimulator()
        self.cfg = config or EngineConfig()
        self.stacker: Ridge | None = None
        self.qbg = qb_games
        self.qb_repl = P.replacement_level(qb_games) if qb_games is not None else 0.0

    # ---- ratings ---------------------------------------------------------
    def _season_prior(self, games: pd.DataFrame, eff: pd.DataFrame, season: int):
        prev = games[(games["season"] == season - 1) & games["result"].notna()]
        if prev.empty:
            return None, None, None
        pr = R.fit_points_ratings(prev, half_life_days=self.cfg.half_life_days)
        pr = R.regress_to_prior(pr, self.cfg.prior_keep)
        pe = eff[eff["season"] == season - 1]
        er = R.fit_efficiency_ratings(pe, prior_plays=50, half_life_days=self.cfg.half_life_days)
        k = self.cfg.prior_keep
        return pr, (er["adj_off"] * k).to_dict(), (er["adj_def"] * k).to_dict()

    def market_ratings_asof(self, games: pd.DataFrame, season: int, week: int) -> R.TeamRatings | None:
        """Market-implied ratings: the same least-squares system solved on past
        closing lines instead of scores. Captures what bettors and bookmakers
        collectively knew (injuries, news) about each team before this week."""
        if "mkt_home_pts" not in games:
            return None
        kw = dict(home_pts_col="mkt_home_pts", away_pts_col="mkt_away_pts", half_life_days=None)
        prev = games[(games["season"] == season - 1)].dropna(subset=["mkt_home_pts"])
        prior = R.regress_to_prior(R.fit_points_ratings(prev, **kw), 0.7) if len(prev) else None
        cur = games[(games["season"] == season) & (games["week"] < week)].dropna(subset=["mkt_home_pts"])
        if cur.empty:
            return prior
        return R.fit_points_ratings(cur, prior=prior, prior_games=3.0, **kw)

    def ratings_asof(self, games: pd.DataFrame, eff: pd.DataFrame, season: int, week: int,
                     prior_adjustments: dict[str, float] | None = None):
        """Ratings using only information available before ``season``/``week``."""
        pr, po, pd_ = self._season_prior(games, eff, season)
        if pr is not None and prior_adjustments:
            pr = R.adjust_prior(pr, prior_adjustments)
        cur = games[(games["season"] == season) & (games["week"] < week) & games["result"].notna()]
        if cur.empty:
            pts = pr
        else:
            pts = R.fit_points_ratings(cur, prior=pr, prior_games=self.cfg.prior_games,
                                       half_life_days=self.cfg.half_life_days)
        ce = eff[eff["game_id"].isin(cur["game_id"])]
        if ce.empty:
            er = pd.DataFrame({"adj_off": pd.Series(po or {}), "adj_def": pd.Series(pd_ or {})})
        else:
            er = R.fit_efficiency_ratings(ce, po, pd_, prior_plays=self.cfg.epa_prior_plays,
                                          half_life_days=self.cfg.half_life_days)
        return pts, er

    # ---- features --------------------------------------------------------
    def side_features(self, g: pd.DataFrame, pts: R.TeamRatings, er: pd.DataFrame,
                      season: int | None = None, week: int | None = None,
                      qb_overrides: dict[str, str] | None = None,
                      mkt: R.TeamRatings | None = None) -> pd.DataFrame:
        """Two rows per game (home side, away side) of stacker features."""
        qb = {}
        if self.qbg is not None and season is not None:
            qb = P.qb_edges(g, self.qbg, season, week, self.qb_repl, qb_overrides)
        rows = []
        for _, r in g.iterrows():
            hp, ap = pts.expected_points(r["home_team"], r["away_team"], bool(r["neutral"]))
            mh, ma = (mkt or pts).expected_points(r["home_team"], r["away_team"], bool(r["neutral"]))
            off = er["adj_off"].to_dict() if len(er) else {}
            de = er["adj_def"].to_dict() if len(er) else {}
            for side, team, opp, p, mp, home, rest in (
                ("home", r["home_team"], r["away_team"], hp, mh, 1, r["home_rest"] - r["away_rest"]),
                ("away", r["away_team"], r["home_team"], ap, ma, 0, r["away_rest"] - r["home_rest"]),
            ):
                rows.append({"game_id": r["game_id"], "side": side, "team": team, "opp": opp,
                             "pts_pred": p, "mkt_pred": mp, "epa_pred": off.get(team, 0.0) - de.get(opp, 0.0),
                             "qb_edge": qb.get((r["game_id"], team), 0.0),
                             "opp_qb_edge": qb.get((r["game_id"], opp), 0.0),
                             "is_home": 0 if r["neutral"] else home,
                             "rest_edge": float(np.clip(rest, -7, 7)), "wind": float(r["wind"]),
                             "is_dome": float(r["is_dome"]), "early": float(r["week"] <= 4)})
        return pd.DataFrame(rows)

    def walk_forward_features(self, games: pd.DataFrame, eff: pd.DataFrame, seasons) -> pd.DataFrame:
        """Out-of-sample features for every game: each week uses only prior data."""
        out = []
        for s in seasons:
            gs = games[games["season"] == s]
            for w in sorted(gs["week"].unique()):
                pts, er = self.ratings_asof(games, eff, s, w)
                if pts is None:
                    continue
                mkt = self.market_ratings_asof(games, s, w)
                f = self.side_features(gs[gs["week"] == w], pts, er, s, w, mkt=mkt)
                f["season"], f["week"] = s, w
                out.append(f)
        return pd.concat(out, ignore_index=True)

    # ---- stacker ---------------------------------------------------------
    def fit_stacker(self, feats: pd.DataFrame, games: pd.DataFrame) -> "GameEngine":
        y = self._actual_points(feats, games)
        ok = y.notna()
        self.stacker = Ridge(alpha=self.cfg.ridge_alpha).fit(feats.loc[ok, STACK_FEATURES], y[ok])
        return self

    @staticmethod
    def _actual_points(feats, games):
        g = games.set_index("game_id")
        return pd.Series(np.where(feats["side"] == "home",
                                  feats["game_id"].map(g["home_score"]),
                                  feats["game_id"].map(g["away_score"])), index=feats.index)

    def expected_points(self, feats: pd.DataFrame) -> pd.Series:
        if self.stacker is None:
            return feats["pts_pred"]
        return pd.Series(self.stacker.predict(feats[STACK_FEATURES]), index=feats.index)

    def game_projections(self, feats: pd.DataFrame) -> pd.DataFrame:
        f = feats.assign(exp_pts=self.expected_points(feats))
        h = f[f["side"] == "home"].set_index("game_id")
        a = f[f["side"] == "away"].set_index("game_id")
        return pd.DataFrame({"home_team": h["team"], "away_team": a["team"],
                             "home_exp": h["exp_pts"], "away_exp": a["exp_pts"]})

    def simulate(self, home_exp: float, away_exp: float, n_sims: int = 20000) -> SimResult:
        return self.sim.simulate(home_exp, away_exp, n_sims=n_sims)


# ---------------------------------------------------------------- market blend

def fit_market_blend(model: np.ndarray, market: np.ndarray, actual: np.ndarray) -> float:
    """Best weight w in [0, 1] for  actual ~ (1-w)*market + w*model.

    This measures how much the model knows that the market doesn't. Against
    closing NFL lines it comes out small (a few percent), and that is expected:
    closing lines are very efficient. The weight sets how far betting advice
    moves away from the market.
    """
    d = np.asarray(model) - np.asarray(market)
    r = np.asarray(actual) - np.asarray(market)
    w = float(np.dot(d, r) / max(np.dot(d, d), 1e-9))
    return float(np.clip(w, 0.0, 1.0))


def blend_projection(model_home: float, model_away: float, market_spread: float | None,
                     market_total: float | None, w_margin: float, w_total: float) -> tuple[float, float]:
    """Combine model and market into betting-grade expected points.

    ``market_spread`` is the expected home margin (positive = home favoured),
    ``market_total`` the posted total. Missing market data -> pure model.
    """
    margin = model_home - model_away
    total = model_home + model_away
    if market_spread is not None and not np.isnan(market_spread):
        margin = (1 - w_margin) * market_spread + w_margin * margin
    if market_total is not None and not np.isnan(market_total):
        total = (1 - w_total) * market_total + w_total * total
    return (total + margin) / 2, (total - margin) / 2
