"""Turn model probabilities + sportsbook prices into betting advice."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .odds import expected_value, implied_prob, kelly_fraction
from .simulate import SimResult


@dataclass
class Bet:
    game: str
    market: str        # spread / total / moneyline
    selection: str     # e.g. "KC -3.5", "Over 47.5", "BUF ML"
    price: float       # American odds
    model_prob: float  # model's win probability (pushes excluded)
    break_even: float  # probability needed to profit at this price
    edge: float        # model_prob - break_even
    ev_per_unit: float
    kelly_stake: float  # share of bankroll (fractional Kelly)
    grade: str

    def dict(self) -> dict:
        return asdict(self)


def _grade(edge: float) -> str:
    return "A" if edge >= 0.08 else "B" if edge >= 0.05 else "C" if edge >= 0.03 else "pass"


def _bet(game, market, selection, price, win, push, kelly_fraction_, p_std, max_stake):
    # a push refunds the stake, so price the bet on non-push outcomes
    p = win / max(1.0 - push, 1e-9)
    be = implied_prob(price)
    edge = p - be
    return Bet(game, market, selection, price, p, be, edge, expected_value(p, price),
               kelly_fraction(p, price, kelly_fraction_, p_std, max_stake), _grade(edge))


def evaluate_markets(sim: SimResult, home: str, away: str, *, home_spread=None,
                     home_spread_price=-110, away_spread_price=-110, total=None,
                     over_price=-110, under_price=-110, home_ml=None, away_ml=None,
                     kelly_fraction_: float = 0.25, p_std: float = 0.02,
                     max_stake: float = 0.03) -> list[Bet]:
    """Every side of every market offered, scored by the model."""
    game = f"{away} @ {home}"
    out: list[Bet] = []
    kw = dict(kelly_fraction_=kelly_fraction_, p_std=p_std, max_stake=max_stake)
    if home_spread is not None:
        c = sim.p_home_cover(home_spread)
        out.append(_bet(game, "spread", f"{home} {home_spread:+g}", home_spread_price, c["win"], c["push"], **kw))
        out.append(_bet(game, "spread", f"{away} {-home_spread:+g}", away_spread_price, c["lose"], c["push"], **kw))
    if total is not None:
        t = sim.p_over(total)
        out.append(_bet(game, "total", f"Over {total:g}", over_price, t["over"], t["push"], **kw))
        out.append(_bet(game, "total", f"Under {total:g}", under_price, t["under"], t["push"], **kw))
    if home_ml is not None and away_ml is not None:
        ph = sim.p_home_win()
        out.append(_bet(game, "moneyline", f"{home} ML", home_ml, ph, 0.0, **kw))
        out.append(_bet(game, "moneyline", f"{away} ML", away_ml, 1 - ph, 0.0, **kw))
    return out


def recommend(bets: list[Bet], min_edge: float = 0.03, max_dog_price: float = 250) -> list[Bet]:
    """Only bets with a real edge, best first.

    Moneyline underdogs longer than ``max_dog_price`` are skipped: in the
    2023-2025 backtest these "edges" were mostly noise and lost money.
    """
    return sorted([b for b in bets if b.edge >= min_edge and b.kelly_stake > 0
                   and not (b.market == "moneyline" and b.price > max_dog_price)],
                  key=lambda b: -b.edge)
