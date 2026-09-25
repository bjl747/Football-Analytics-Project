import numpy as np
import pytest

from superalgo.advice import evaluate_markets, recommend
from superalgo.simulate import GameSimulator, GameState


@pytest.fixture(scope="module")
def sim():
    return GameSimulator(seed=11)


def test_means_hit_targets(sim):
    r = sim.simulate(27.0, 20.0, n_sims=30000)
    s = r.summary()
    assert s["home_points"] == pytest.approx(27.0, abs=0.8)
    assert s["away_points"] == pytest.approx(20.0, abs=0.8)
    assert 0.65 < s["p_home_win"] < 0.80


def test_probabilities_are_consistent(sim):
    r = sim.simulate(24.0, 21.0, n_sims=20000)
    c = r.p_home_cover(-3.0)
    assert c["win"] + c["push"] + c["lose"] == pytest.approx(1.0)
    assert c["push"] > 0.05  # 3 is a key number
    o = r.p_over(44.5)
    assert o["over"] + o["under"] == pytest.approx(1.0)
    assert sum(p for *_, p in r.most_likely_scores(50)) <= 1.0


def test_key_numbers_three_and_seven(sim):
    r = sim.simulate(23.0, 22.0, n_sims=40000)
    md = r.margin_distribution(-10, 10)
    assert md[3] > md[2] and md[3] > md[4]
    assert md[7] > md[5]


def test_live_state_blowout(sim):
    r = sim.simulate(24, 21, n_sims=5000,
                     state=GameState(home_score=35, away_score=3, drives_remaining=2, home_has_ball=True))
    assert r.p_home_win() > 0.99


def test_advice_finds_mispriced_line(sim):
    r = sim.simulate(30.0, 17.0, n_sims=20000)
    bets = evaluate_markets(r, "HOME", "AWAY", home_spread=-3.0, total=47.0, home_ml=-150, away_ml=130)
    rec = recommend(bets)
    assert rec and rec[0].selection in ("HOME -3", "HOME ML")
    assert all(0 < b.kelly_stake <= 0.03 for b in rec)
