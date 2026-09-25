"""Dynamic opponent-adjusted team ratings via a Kalman filter.

Standard least-squares ratings treat a team as a fixed number inside a
window. Real teams change week to week (injuries, scheme, a QB learning).
Here each team's offence and defence rating for any statistic is a hidden
state that drifts as a random walk:

    week to week:     rating_t = rating_{t-1} + noise(q_week)
    season to season: rating   = carry * rating + noise(q_season)   (regression to mean)
    each game:        stat(team vs opp) = mu + off_team - def_opp + hfa*home + noise(r / plays)

The Kalman filter gives the exact Bayesian update after every game, plus an
*uncertainty* for every rating. The update is walk-forward by construction:
a rating only ever contains games already played. The 4 hyperparameters
(q_week, q_season, carry, r) are fitted per statistic by maximising the
likelihood of the next week's games, so each stat learns its own stability
(e.g. passing efficiency is stickier than fumble recoveries).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanParams:
    q_week: float
    q_season: float
    carry: float
    r: float          # observation noise variance for a game of `ref_weight`
    prior_var: float  # initial rating variance
    ref_weight: float = 1.0


def run_filter(obs: pd.DataFrame, value: str, params: KalmanParams,
               weight: str | None = None, return_loglik: bool = False):
    """Run the filter over team-game observations.

    ``obs`` needs columns: game_id, season, week, team, opp, is_home, <value>.
    Returns a DataFrame keyed by (game_id, team) with the PRE-GAME prediction
    ``pred`` (expected value of the stat for this team's offence in this game),
    its variance ``pred_var``, and pre-game off/def ratings.
    """
    obs = obs.sort_values(["season", "week", "game_id"])
    teams = sorted(set(obs["team"]) | set(obs["opp"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    k = 2 * n + 2
    MU, HFA = 2 * n, 2 * n + 1
    x = np.zeros(k)
    x[MU] = obs[value].iloc[: min(len(obs), 500)].mean()
    P = np.eye(k) * params.prior_var
    P[MU, MU] = params.prior_var
    P[HFA, HFA] = params.prior_var
    team_dims = np.arange(2 * n)

    ti = obs["team"].map(idx).to_numpy()
    oi = obs["opp"].map(idx).to_numpy()
    home = obs["is_home"].to_numpy(float)
    y = obs[value].to_numpy(float)
    w = obs[weight].to_numpy(float) / params.ref_weight if weight else np.ones(len(obs))
    seen = ~(np.isnan(y) | np.isnan(w))  # unplayed games: predict only, no update
    seasons = obs["season"].to_numpy()
    weeks = obs["week"].to_numpy()

    pred = np.empty(len(obs)); pvar = np.empty(len(obs))
    off_pre = np.empty(len(obs)); def_pre = np.empty(len(obs))
    loglik = 0.0
    cur_s, cur_w = None, None
    i = 0
    N = len(obs)
    while i < N:
        s, wk = seasons[i], weeks[i]
        # time update
        if cur_s is not None:
            if s != cur_s:
                x[team_dims] *= params.carry
                P[np.ix_(team_dims, team_dims)] *= params.carry ** 2
                P[team_dims, team_dims] += params.q_season
                P[MU, MU] += params.q_season * 0.1
            elif wk != cur_w:
                P[team_dims, team_dims] += params.q_week
        cur_s, cur_w = s, wk
        j = i
        while j < N and seasons[j] == s and weeks[j] == wk:
            j += 1
        # record pre-week predictions for all games this week
        for m in range(i, j):
            a_idx = (ti[m], n + oi[m], MU, HFA)
            a_val = (1.0, -1.0, 1.0, home[m])
            pred[m] = sum(v * x[c] for c, v in zip(a_idx, a_val))
            sub = P[np.ix_(a_idx, a_idx)]
            av = np.array(a_val)
            pvar[m] = av @ sub @ av
            off_pre[m] = x[ti[m]]; def_pre[m] = x[n + oi[m]]
        # measurement updates
        for m in range(i, j):
            if not seen[m]:
                continue
            a = np.zeros(k)
            a[ti[m]] = 1.0; a[n + oi[m]] = -1.0; a[MU] = 1.0; a[HFA] = home[m]
            R = params.r / max(w[m], 1e-3)
            Pa = P @ a
            S = a @ Pa + R
            resid = y[m] - a @ x
            if return_loglik:
                loglik += -0.5 * (np.log(2 * np.pi * S) + resid ** 2 / S)
            K = Pa / S
            x += K * resid
            P -= np.outer(K, Pa)
        i = j
    out = pd.DataFrame({"game_id": obs["game_id"].to_numpy(), "team": obs["team"].to_numpy(),
                        "pred": pred, "pred_var": pvar, "off": off_pre, "def": def_pre})
    if return_loglik:
        return out, loglik
    return out


def fit_params(obs: pd.DataFrame, value: str, weight: str | None = None,
               seasons_fit=None, init: KalmanParams | None = None) -> KalmanParams:
    """Maximise next-game predictive likelihood over the hyperparameters."""
    from scipy.optimize import minimize
    o = obs.dropna(subset=[value])
    v = o[value].var()
    ref = o[weight].mean() if weight else 1.0
    init = init or KalmanParams(q_week=v * 0.002, q_season=v * 0.02, carry=0.6, r=v * 0.8,
                                prior_var=v * 0.1, ref_weight=ref)

    def unpack(z):
        return KalmanParams(q_week=np.exp(z[0]), q_season=np.exp(z[1]),
                            carry=1 / (1 + np.exp(-z[2])), r=np.exp(z[3]),
                            prior_var=init.prior_var, ref_weight=ref)

    def nll(z):
        # only score seasons after the first (burn-in) to avoid cold-start noise
        p = unpack(z)
        res, _ = run_filter(o, value, p, weight, return_loglik=True)
        res = res.assign(y=o.sort_values(["season", "week", "game_id"])[value].to_numpy(),
                         season=o.sort_values(["season", "week", "game_id"])["season"].to_numpy())
        if seasons_fit is not None:
            res = res[res["season"].isin(seasons_fit)]
        S = res["pred_var"] + p.r
        return float(np.sum(0.5 * (np.log(S) + (res["y"] - res["pred"]) ** 2 / S)))

    z0 = np.array([np.log(init.q_week), np.log(init.q_season),
                   np.log(init.carry / (1 - init.carry)), np.log(init.r)])
    r = minimize(nll, z0, method="Nelder-Mead", options={"maxiter": 250, "xatol": 1e-3, "fatol": 1e-2})
    return unpack(r.x)
