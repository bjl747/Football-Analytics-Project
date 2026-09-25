import numpy as np
import pandas as pd
import pytest

from superalgo import ratings as R


def _synthetic(n_teams=12, rounds=6, hfa=2.5, seed=0):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    off = dict(zip(teams, rng.normal(0, 4, n_teams)))
    de = dict(zip(teams, rng.normal(0, 4, n_teams)))
    rows = []
    for r in range(rounds):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            hp = 22 + off[h] - de[a] + hfa / 2 + rng.normal(0, 1)
            ap = 22 + off[a] - de[h] - hfa / 2 + rng.normal(0, 1)
            rows.append({"home_team": h, "away_team": a, "home_score": hp, "away_score": ap,
                         "result": hp - ap, "gameday": f"2024-09-{r + 1:02d}", "neutral": False})
    return pd.DataFrame(rows), off, de, hfa


def test_points_ratings_recover_truth():
    g, off, de, hfa = _synthetic(rounds=30)
    r = R.fit_points_ratings(g, half_life_days=None)
    true_net = pd.Series({t: off[t] + de[t] for t in off})
    est_net = pd.Series(r.net)
    true_net -= true_net.mean()
    assert np.corrcoef(true_net, est_net[true_net.index])[0, 1] > 0.98
    assert r.hfa == pytest.approx(hfa, abs=1.0)
    true_avg = 22 + np.mean(list(off.values())) - np.mean(list(de.values()))
    assert r.league_avg == pytest.approx(true_avg, abs=0.5)


def test_prior_dominates_with_few_games():
    g, *_ = _synthetic(rounds=1)
    prior = R.TeamRatings(sorted(set(g.home_team) | set(g.away_team)),
                          {t: 10.0 if t == "T0" else 0.0 for t in set(g.home_team) | set(g.away_team)},
                          {t: 0.0 for t in set(g.home_team) | set(g.away_team)}, 2.0, 22.0)
    weak = R.fit_points_ratings(g, prior=prior, prior_games=100, half_life_days=None)
    assert weak.offense["T0"] > 7  # strong prior barely moves after one game


def test_massey_normal_equation_and_colley():
    g, off, de, hfa = _synthetic(rounds=30)
    m, h = R.massey(g, margin_col="result")
    assert abs(m.sum()) < 1e-8
    assert h == pytest.approx(hfa, abs=1.0)
    c = R.colley(g)
    assert c.mean() == pytest.approx(0.5, abs=1e-9)  # Colley ratings average 1/2
    assert np.corrcoef(m[c.index], c)[0, 1] > 0.7
