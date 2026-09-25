"""Walk-forward backtest against real closing lines.

Every prediction uses only data available before that week. The market
comparison uses nflverse closing lines (spread, total, moneyline and their
prices). Beating the *closing* line is the hardest test there is: professional
bettors aim to beat it, and most models never do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .advice import evaluate_markets, recommend
from .engine import GameEngine
from .odds import devig


def simulate_games(engine: GameEngine, proj: pd.DataFrame, n_sims: int = 10000) -> pd.DataFrame:
    rows = []
    for gid, r in proj.iterrows():
        sim = engine.simulate(r["home_exp"], r["away_exp"], n_sims=n_sims)
        rows.append({"game_id": gid, "sim": sim, **sim.summary()})
    return pd.DataFrame(rows).set_index("game_id")


def evaluate(engine: GameEngine, proj: pd.DataFrame, games: pd.DataFrame,
             sims: pd.DataFrame, min_edge: float = 0.03) -> dict:
    g = games.set_index("game_id").loc[proj.index]
    df = proj.join(sims.drop(columns=["sim"])).join(
        g[["season", "week", "result", "total", "spread_line", "total_line", "home_moneyline",
           "away_moneyline", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds"]])
    df = df.dropna(subset=["result", "spread_line", "total_line"])
    model_margin = df["home_exp"] - df["away_exp"]

    out = {
        "games": int(len(df)),
        "margin_mae_model": float((model_margin - df["result"]).abs().mean()),
        "margin_mae_market": float((df["spread_line"] - df["result"]).abs().mean()),
        "total_mae_model": float(((df["home_exp"] + df["away_exp"]) - df["total"]).abs().mean()),
        "total_mae_market": float((df["total_line"] - df["total"]).abs().mean()),
        "corr_model_vs_market_spread": float(np.corrcoef(model_margin, df["spread_line"])[0, 1]),
    }
    ml = df.dropna(subset=["home_moneyline", "away_moneyline"])
    ml = ml[ml["result"] != 0]
    won = (ml["result"] > 0).astype(float)
    mkt = np.array([devig(h, a)[0] for h, a in zip(ml["home_moneyline"], ml["away_moneyline"])])
    out["brier_model"] = float(np.mean((ml["p_home_win"] - won) ** 2))
    out["brier_market"] = float(np.mean((mkt - won) ** 2))
    out["logloss_model"] = float(-np.mean(won * np.log(ml["p_home_win"].clip(1e-4, 1 - 1e-4))
                                          + (1 - won) * np.log((1 - ml["p_home_win"]).clip(1e-4, 1 - 1e-4))))
    out["logloss_market"] = float(-np.mean(won * np.log(mkt) + (1 - won) * np.log(1 - mkt)))

    # betting simulation at closing prices, flat 1 unit and quarter-Kelly
    ledger = []
    for gid, r in df.iterrows():
        sim = sims.loc[gid, "sim"]
        bets = evaluate_markets(
            sim, r["home_team"], r["away_team"], home_spread=-r["spread_line"],
            home_spread_price=_price(r["home_spread_odds"]), away_spread_price=_price(r["away_spread_odds"]),
            total=r["total_line"], over_price=_price(r["over_odds"]), under_price=_price(r["under_odds"]),
            home_ml=r["home_moneyline"] if pd.notna(r["home_moneyline"]) else None,
            away_ml=r["away_moneyline"] if pd.notna(r["away_moneyline"]) else None)
        for b in recommend(bets, min_edge):
            ledger.append({"game_id": gid, "season": r["season"], "week": r["week"], **b.dict(),
                           "outcome": _settle(b, r)})
    led = pd.DataFrame(ledger)
    out["bets"] = _ledger_stats(led)
    return {"summary": out, "ledger": led, "frame": df}


def _price(x) -> float:
    return float(x) if pd.notna(x) and abs(x) >= 100 else -110.0


def _settle(b, r) -> float:
    """Profit per 1 unit staked (0 on push)."""
    from .odds import american_to_decimal
    home, away = r["home_team"], r["away_team"]
    margin, total = r["result"], r["total"]
    sel = b.selection
    if b.market == "spread":
        line = float(sel.split()[-1])
        m = margin if sel.startswith(home) else -margin
        res = np.sign(m + line)
    elif b.market == "total":
        line = float(sel.split()[-1])
        res = np.sign(total - line) * (1 if sel.startswith("Over") else -1)
    else:
        m = margin if sel.startswith(home) else -margin
        res = np.sign(m)
    if res == 0:
        return 0.0
    return american_to_decimal(b.price) - 1.0 if res > 0 else -1.0


def _ledger_stats(led: pd.DataFrame) -> dict:
    if led.empty:
        return {"n": 0}
    stats = {}
    for market, d in [("all", led)] + list(led.groupby("market")):
        decided = d[d["outcome"] != 0]
        stats[market] = {
            "n": int(len(d)),
            "win_rate": float((decided["outcome"] > 0).mean()) if len(decided) else None,
            "roi_flat": float(d["outcome"].mean()),
            "units_flat": float(d["outcome"].sum()),
            "kelly_growth": float(np.prod(1 + d["kelly_stake"] * d["outcome"]) - 1),
        }
    return stats
