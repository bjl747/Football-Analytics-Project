"""Power ratings: least-squares (Massey) systems with Bayesian priors, Colley,
and opponent-adjusted efficiency ratings built on garbage-time-filtered EPA.

Core maths: each game gives one equation, e.g. rating_home - rating_away + HFA
= margin. With many more games than teams the system Ax = b is overdetermined,
so we take the least-squares solution from the normal equation

    x_hat = (A^T W A + L)^-1 (A^T W b + L mu)

W   = recency weights (recent games count more),
L   = prior strength (how many "pseudo-games" the preseason prior is worth),
mu  = the Bayesian preseason prior.
With L = 0 and W = I this is exactly x_hat = (A^T A)^-1 A^T b. The prior's
influence decays automatically as real games pile up in A^T W A.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _recency_weights(dates: pd.Series, half_life_days: float | None) -> np.ndarray:
    if not half_life_days:
        return np.ones(len(dates))
    d = pd.to_datetime(dates)
    age = (d.max() - d).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age / half_life_days)


@dataclass
class TeamRatings:
    """Offence/defence point ratings on top of a league-average score."""
    teams: list[str]
    offense: dict[str, float]
    defense: dict[str, float]   # positive = good defence (allows fewer points)
    hfa: float                  # home-field advantage in points (margin)
    league_avg: float           # average points per team per game
    extra: dict = field(default_factory=dict)

    @property
    def net(self) -> dict[str, float]:
        return {t: self.offense[t] + self.defense[t] for t in self.teams}

    def expected_points(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        h = 0.0 if neutral else self.hfa / 2
        home_pts = self.league_avg + self.offense.get(home, 0) - self.defense.get(away, 0) + h
        away_pts = self.league_avg + self.offense.get(away, 0) - self.defense.get(home, 0) - h
        return home_pts, away_pts

    def table(self) -> pd.DataFrame:
        df = pd.DataFrame({"team": self.teams,
                           "offense": [self.offense[t] for t in self.teams],
                           "defense": [self.defense[t] for t in self.teams]})
        df["net"] = df["offense"] + df["defense"]
        return df.sort_values("net", ascending=False).reset_index(drop=True)


def fit_points_ratings(games: pd.DataFrame, prior: TeamRatings | None = None,
                       prior_games: float = 4.0, half_life_days: float | None = 120,
                       home_col: str = "home_team", away_col: str = "away_team",
                       home_pts_col: str = "home_score", away_pts_col: str = "away_score",
                       neutral_col: str | None = "neutral", date_col: str = "gameday",
                       ridge: float = 1e-3) -> TeamRatings:
    """Bayesian least-squares offence/defence ratings from game scores.

    Unknowns: [off_1..off_n, def_1..def_n, hfa, league_avg]. Each game adds two
    rows (one per team's points scored).
    """
    g = games.dropna(subset=[home_pts_col, away_pts_col])
    teams = sorted(set(g[home_col]) | set(g[away_col]) | (set(prior.teams) if prior else set()))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    k = 2 * n + 2
    HFA, AVG = 2 * n, 2 * n + 1

    m = len(g)
    A = np.zeros((2 * m, k))
    b = np.zeros(2 * m)
    neutral = g[neutral_col].astype(bool).to_numpy() if neutral_col and neutral_col in g else np.zeros(m, bool)
    hi = g[home_col].map(idx).to_numpy()
    ai = g[away_col].map(idx).to_numpy()
    r = np.arange(m)
    # home points = avg + off_home - def_away + hfa/2
    A[2 * r, hi] = 1; A[2 * r, n + ai] = -1; A[2 * r, AVG] = 1
    A[2 * r, HFA] = np.where(neutral, 0, 0.5)
    b[2 * r] = g[home_pts_col].to_numpy(float)
    # away points = avg + off_away - def_home - hfa/2
    A[2 * r + 1, ai] = 1; A[2 * r + 1, n + hi] = -1; A[2 * r + 1, AVG] = 1
    A[2 * r + 1, HFA] = np.where(neutral, 0, -0.5)
    b[2 * r + 1] = g[away_pts_col].to_numpy(float)

    w = np.repeat(_recency_weights(g[date_col], half_life_days) if date_col in g else np.ones(m), 2)

    mu = np.zeros(k)
    L = np.full(k, ridge)
    if prior is not None:
        for t, i in idx.items():
            mu[i] = prior.offense.get(t, 0.0)
            mu[n + i] = prior.defense.get(t, 0.0)
        L[: 2 * n] = prior_games
        mu[HFA], mu[AVG] = prior.hfa, prior.league_avg
        L[HFA] = L[AVG] = 5.0
    else:
        L[: 2 * n] = max(ridge, 0.5)  # light shrinkage so early-season ratings are sane
    AtW = A.T * w
    x = np.linalg.solve(AtW @ A + np.diag(L), AtW @ b + L * mu)

    off, de = x[:n], x[n:2 * n]
    # identifiability: centre offence and defence at zero, fold into league_avg
    off_c, def_c = off - off.mean(), de - de.mean()
    league_avg = x[AVG] + off.mean() - de.mean()
    return TeamRatings(teams, dict(zip(teams, off_c)), dict(zip(teams, def_c)),
                       float(x[HFA]), float(league_avg))


def regress_to_prior(r: TeamRatings, keep: float = 0.6,
                     league_avg: float | None = None, hfa: float | None = None) -> TeamRatings:
    """Preseason prior from last season: shrink ratings toward average.

    ``keep`` = share of last season's rating carried over. In the NFL roughly
    half to two-thirds carries over; college (with recruiting and returning
    production data) can adjust this per team via ``adjust_prior``.
    """
    return TeamRatings(r.teams, {t: v * keep for t, v in r.offense.items()},
                       {t: v * keep for t, v in r.defense.items()},
                       r.hfa if hfa is None else hfa,
                       r.league_avg if league_avg is None else league_avg)


def adjust_prior(prior: TeamRatings, adjustments: dict[str, float]) -> TeamRatings:
    """Add per-team net point adjustments to a prior (e.g. from returning
    production, recruiting composites, QB changes or market win totals).
    Split evenly between offence and defence."""
    off = dict(prior.offense); de = dict(prior.defense)
    for t, v in adjustments.items():
        off[t] = off.get(t, 0.0) + v / 2
        de[t] = de.get(t, 0.0) + v / 2
    teams = sorted(set(prior.teams) | set(adjustments))
    for t in teams:
        off.setdefault(t, 0.0); de.setdefault(t, 0.0)
    return TeamRatings(teams, off, de, prior.hfa, prior.league_avg)


def massey(games: pd.DataFrame, home_col="home_team", away_col="away_team",
           margin_col="result", neutral_col: str | None = "neutral",
           cap: float | None = None) -> tuple[pd.Series, float]:
    """Classic Massey margin ratings via the normal equation.

    Returns (ratings, home-field advantage). ``cap`` limits blowout margins.
    """
    g = games.dropna(subset=[margin_col])
    teams = sorted(set(g[home_col]) | set(g[away_col]))
    idx = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(g)
    A = np.zeros((m + 1, n + 1))
    r = np.arange(m)
    A[r, g[home_col].map(idx)] = 1
    A[r, g[away_col].map(idx)] = -1
    neutral = g[neutral_col].astype(bool).to_numpy() if neutral_col and neutral_col in g else np.zeros(m, bool)
    A[r, n] = np.where(neutral, 0, 1)
    b = np.zeros(m + 1)
    marg = g[margin_col].to_numpy(float)
    b[:m] = np.clip(marg, -cap, cap) if cap else marg
    A[m, :n] = 1  # ratings sum to zero (makes A^T A invertible)
    x = np.linalg.solve(A.T @ A, A.T @ b)
    return pd.Series(x[:n], index=teams).sort_values(ascending=False), float(x[n])


def colley(games: pd.DataFrame, home_col="home_team", away_col="away_team",
           margin_col="result") -> pd.Series:
    """Colley Matrix ratings: wins and losses only, margin ignored.

    Solves C r = b with C_ii = 2 + games_i, C_ij = -games(i,j),
    b_i = 1 + (wins_i - losses_i) / 2. Ties count half.
    """
    g = games.dropna(subset=[margin_col])
    teams = sorted(set(g[home_col]) | set(g[away_col]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    C = 2 * np.eye(n)
    b = np.ones(n)
    for h, a, mg in zip(g[home_col].map(idx), g[away_col].map(idx), g[margin_col]):
        C[h, h] += 1; C[a, a] += 1; C[h, a] -= 1; C[a, h] -= 1
        s = 0.5 if mg > 0 else (-0.5 if mg < 0 else 0.0)
        b[h] += s; b[a] -= s
    return pd.Series(np.linalg.solve(C, b), index=teams).sort_values(ascending=False)


# ---------------------------------------------------------------- efficiency

def team_game_efficiency(plays: pd.DataFrame, epa_col: str = "my_epa",
                         wp_col: str = "my_wp", wp_bounds=(0.01, 0.99)) -> pd.DataFrame:
    """Offensive EPA/play for each team in each game, with garbage time removed.

    ``wp_bounds`` is the dual-state "pure strength" filter: plays where the
    offence's win probability is outside the bounds are dropped. The research
    brief specifies 1%/99%; many public models use a tighter 10%/90%.
    """
    p = plays[plays["play_type"].isin(["pass", "run"]) & plays[epa_col].notna()]
    if wp_col in p:
        lo, hi = wp_bounds
        p = p[(p[wp_col] >= lo) & (p[wp_col] <= hi)]
    agg = p.groupby(["game_id", "posteam", "defteam"]).agg(
        epa_play=(epa_col, "mean"), plays=(epa_col, "size"),
        pass_rate=("play_type", lambda s: (s == "pass").mean()),
    ).reset_index()
    first = plays.groupby("game_id")[["home_team", "gameday" if "gameday" in plays else "game_date"]].first()
    first.columns = ["home_team", "date"]
    agg = agg.join(first, on="game_id")
    agg["is_home"] = (agg["posteam"] == agg["home_team"]).astype(float)
    return agg


def fit_efficiency_ratings(eff: pd.DataFrame, prior_off: dict | None = None,
                           prior_def: dict | None = None, prior_plays: float = 150.0,
                           half_life_days: float | None = 120) -> pd.DataFrame:
    """Opponent-adjusted offensive and defensive EPA/play (least squares).

    epa_play(team vs opp) = avg + off_team - def_opp + hfa*is_home, each row
    weighted by snaps. Returns one row per team: adj_off, adj_def (positive =
    good defence) and adj_net, all in EPA per play.
    """
    teams = sorted(set(eff["posteam"]) | set(eff["defteam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, m = len(teams), len(eff)
    A = np.zeros((m, 2 * n + 2))
    r = np.arange(m)
    A[r, eff["posteam"].map(idx)] = 1
    A[r, n + eff["defteam"].map(idx)] = -1
    A[r, 2 * n] = eff["is_home"].to_numpy()
    A[r, 2 * n + 1] = 1
    b = eff["epa_play"].to_numpy(float)
    w = eff["plays"].to_numpy(float) * _recency_weights(eff["date"], half_life_days)

    mu = np.zeros(2 * n + 2)
    L = np.full(2 * n + 2, 1e-3)
    L[: 2 * n] = prior_plays
    for t, i in idx.items():
        if prior_off: mu[i] = prior_off.get(t, 0.0)
        if prior_def: mu[n + i] = prior_def.get(t, 0.0)
    AtW = A.T * w
    x = np.linalg.solve(AtW @ A + np.diag(L), AtW @ b + L * mu)
    off = x[:n] - x[:n].mean()
    de = x[n:2 * n] - x[n:2 * n].mean()
    out = pd.DataFrame({"team": teams, "adj_off": off, "adj_def": de})
    out["adj_net"] = out["adj_off"] + out["adj_def"]
    return out.set_index("team")
