"""Odds conversion, vig removal and Fractional Kelly bet sizing."""
from __future__ import annotations

import numpy as np


def american_to_decimal(american: float) -> float:
    american = float(american)
    if american >= 100:
        return 1.0 + american / 100.0
    if american <= -100:
        return 1.0 + 100.0 / -american
    raise ValueError("American odds must be <= -100 or >= +100")


def decimal_to_american(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be > 1")
    return (decimal - 1.0) * 100.0 if decimal >= 2.0 else -100.0 / (decimal - 1.0)


def implied_prob(american: float) -> float:
    """Break-even probability of a price (includes the bookmaker's vig)."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> float:
    """Fair (no-vig) American price for probability ``p``."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    return decimal_to_american(1.0 / p)


def devig(*american_prices: float) -> np.ndarray:
    """Remove the bookmaker margin proportionally, returning fair probabilities."""
    raw = np.array([implied_prob(a) for a in american_prices])
    return raw / raw.sum()


def kelly_fraction(p: float, american: float, fraction: float = 0.25,
                   p_std: float = 0.0, max_stake: float = 0.05) -> float:
    """Stake as a share of bankroll using Fractional Kelly.

    Full Kelly is f* = (b*p - q) / b, where b is net decimal odds and q = 1 - p.
    Betting full Kelly on noisy probability estimates over-bets and suffers
    "negative geometric drag" (volatility eats compound growth), so we:

    * shrink ``p`` by ``p_std`` (one standard error of the model's probability),
      so a less certain edge gets a smaller bet;
    * multiply by ``fraction`` (0.25 = quarter Kelly);
    * cap the stake at ``max_stake`` of bankroll.

    Returns 0 when there is no edge.
    """
    b = american_to_decimal(american) - 1.0
    p_eff = min(max(p - p_std, 0.0), 1.0)
    f_star = (b * p_eff - (1.0 - p_eff)) / b
    if f_star <= 0:
        return 0.0
    return float(min(fraction * f_star, max_stake))


def expected_value(p: float, american: float) -> float:
    """Expected profit per 1 unit staked."""
    b = american_to_decimal(american) - 1.0
    return p * b - (1.0 - p)
