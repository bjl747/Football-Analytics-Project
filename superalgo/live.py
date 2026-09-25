"""Live (in-game) pricing.

Two engines, both driven by the current game state only (Markov property):

* ``live_win_prob``: instant win probability from the WP GAM (microseconds).
* ``live_price``: re-runs the drive-level Monte Carlo from the current score,
  possession and remaining drives, giving live spread / total / moneyline
  probabilities (tens of milliseconds for 10k sims).

Streaming infrastructure (Kafka / Flink / WebSockets) is a deployment layer on
top of these functions. It is deliberately not built yet because it costs money
to run and needs a paid real-time data feed. ``LiveGame`` is the in-process
version: feed it play events and it re-prices after each one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .simulate import GameSimulator, GameState, SimResult


def drives_remaining(seconds_remaining: float, drives_per_game: float = 21.8) -> float:
    """Expected number of drives left for BOTH teams combined."""
    return max(seconds_remaining, 0) / 3600.0 * drives_per_game


def live_price(sim: GameSimulator, home_exp_pts: float, away_exp_pts: float,
               home_score: int, away_score: int, seconds_remaining: float,
               home_has_ball: bool, n_sims: int = 10000,
               yardline_100: float | None = None) -> SimResult:
    """Price the rest of a game from its current state.

    ``yardline_100`` (yards to the goal line for the team with the ball) makes
    the drive in progress use field-position scoring odds.
    """
    dr = drives_remaining(seconds_remaining, sim.cal["drives_per_game_mean"])
    state = GameState(home_score=home_score, away_score=away_score,
                      drives_remaining=max(dr, 1.0 if yardline_100 is not None else 0.0),
                      home_has_ball=home_has_ball, seconds_remaining=seconds_remaining,
                      yardline_100=yardline_100)
    old = sim.use_key_numbers
    sim.use_key_numbers = False  # key-number weights are for full-game pricing
    r = sim.simulate(home_exp_pts, away_exp_pts, n_sims=n_sims, state=state)
    sim.use_key_numbers = old
    return r


@dataclass
class LiveGame:
    """Stateful in-game pricer: call ``update`` with each new play."""
    sim: GameSimulator
    home: str
    away: str
    home_exp_pts: float
    away_exp_pts: float
    history: list = field(default_factory=list)

    def update(self, home_score: int, away_score: int, seconds_remaining: float,
               home_has_ball: bool, n_sims: int = 5000) -> dict:
        r = live_price(self.sim, self.home_exp_pts, self.away_exp_pts, home_score,
                       away_score, seconds_remaining, home_has_ball, n_sims=n_sims)
        snap = {"seconds_remaining": seconds_remaining, "home_score": home_score,
                "away_score": away_score, "p_home_win": r.p_home_win(),
                "exp_final_home": float(r.home.mean()), "exp_final_away": float(r.away.mean())}
        self.history.append(snap)
        return snap

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
