"""Situational adjustments that passed BOTH the 2012-2022 development test
and the 2023-2025 holdout (see docs/RESEARCH_LOG.md, experiment 4).

Sizes are shrunk to about 70% of the measured effect, because rules found by
searching many candidates are usually a bit lucky.
"""
from __future__ import annotations

import numpy as np

RULES = {
    # home favourite coming off a bye: market over-prices the rest edge
    # (dev 54.8% / holdout 59.4% fading them; academic support: Frontiers 2024)
    "home_fav_off_bye": {"margin": -0.6},
    # night games finish under the closing total more often (52.8% / 55.1%)
    "primetime": {"total": -0.4},
}

# forecast-wind adjustment to the total (points), outdoor games only.
# Set from the 2022-2025 archived-forecast test; see RESEARCH_LOG experiment 7.
WIND_TOTAL_ADJ = [(0, 9, 0.0), (10, 14, -1.0), (15, 19, -1.5), (20, 99, -1.0)]


def adjustments(home_rest: float, spread_line: float, kick_hour: int,
                forecast_wind: float | None = None, roof: str | None = None) -> dict:
    """Point adjustments to (home margin, total) for the betting view."""
    adj = {"margin": 0.0, "total": 0.0, "flags": []}
    if home_rest is not None and home_rest >= 13 and spread_line is not None and spread_line > 0:
        adj["margin"] += RULES["home_fav_off_bye"]["margin"]; adj["flags"].append("home_fav_off_bye")
    if kick_hour is not None and kick_hour >= 19:
        adj["total"] += RULES["primetime"]["total"]; adj["flags"].append("primetime")
    if forecast_wind is not None and not np.isnan(forecast_wind) and roof not in ("dome", "closed"):
        for lo, hi, v in WIND_TOTAL_ADJ:
            if lo <= forecast_wind <= hi and v:
                adj["total"] += v; adj["flags"].append(f"wind_{lo}-{hi}mph")
    return adj


def wong_teaser_legs(home: str, away: str, spread_line: float, total_line: float) -> list[dict]:
    """6-point teaser legs that cross both 3 and 7 (the classic 'Wong' teaser).

    Backtest (closing lines): 75.5% of legs won 2012-2022, 74.4% in 2023-2025;
    77.1% / 75.4% when the total was 49 or lower. A 2-team 6-point teaser at
    -120 needs about 73.9% per leg to break even.
    """
    legs = []
    if spread_line is None or np.isnan(spread_line):
        return legs
    s = spread_line  # expected home margin, + = home favoured
    if 7.5 <= s <= 8.5:
        legs.append({"team": home, "teased_line": -(s - 6)})
    if -2.5 <= s <= -1.5:
        legs.append({"team": home, "teased_line": -s + 6})
    if -8.5 <= s <= -7.5:
        legs.append({"team": away, "teased_line": s + 6})
    if 1.5 <= s <= 2.5:
        legs.append({"team": away, "teased_line": s + 6})
    for leg in legs:
        leg["game"] = f"{away} @ {home}"
        leg["low_total"] = bool(total_line is not None and total_line <= 49)
        leg["hist_leg_win"] = 0.771 if leg["low_total"] else 0.755
    return legs


# Experiment 9: when the pure model disagrees with the opening line, the line tends
# to move toward the model before kickoff. Holdout 2024-25 figures, using ONLY
# information available when lines open (no same-week QB/injury news):
#   |gap| >= 1.5 -> 58.8% move our way (+0.46 pts);  |gap| >= 3 -> 62.2% (+0.48 pts)
# When fresh QB/injury news is in the model before the books react, it rises to
# 69.5% / 79.3% (+1.1 / +1.7 pts): a speed edge.
LINE_MOVE_TABLE = [(3.0, 0.62, 0.5), (1.5, 0.59, 0.46)]


def line_move_signal(home: str, away: str, model_margin: float, current_spread: float) -> dict | None:
    """Tell the bettor whether to bet NOW (before the line moves) or wait.

    model_margin / current_spread: expected home margin (+ = home favoured).
    """
    if current_spread is None or np.isnan(current_spread):
        return None
    gap = model_margin - current_spread
    for k, p, pts in LINE_MOVE_TABLE:
        if abs(gap) >= k:
            side = home if gap > 0 else away
            return {"bet_now_on": side, "model_minus_line": round(float(gap), 1),
                    "hist_prob_line_moves_our_way": p, "hist_avg_points_gained": pts,
                    "note": "Edge comes from betting before the line moves; it fades by kickoff."}
    return None
