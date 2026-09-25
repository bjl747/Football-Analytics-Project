"""Weekly NFL predictions (engine v2) -> output/nfl_<season>_week<NN>.json

    python scripts/predict_week.py --season 2026 --week 3

For each game:
  model        : pure Super-Algo projection (Kalman ratings + QB + injuries + rest/travel)
  betting_view : model blended with the market line, plus situational and
                 weather adjustments that passed out-of-sample tests
  advice       : bets with an edge (spread/total/moneyline), grade and stake
  teaser_legs  : Wong teaser candidates
Prices: The Odds API if ODDS_API_KEY is set, otherwise the free nflverse line.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from superalgo import market as M  # noqa: E402
from superalgo.advice import Bet, evaluate_markets, recommend  # noqa: E402
from superalgo.engine import blend_projection  # noqa: E402
from superalgo.odds import prob_to_american  # noqa: E402
from superalgo.news import fetch_injuries, qb_alerts  # noqa: E402
from superalgo.pipeline import GamePredictor  # noqa: E402
from superalgo.simulate import GameSimulator  # noqa: E402
from superalgo.situational import adjustments, line_move_signal, wong_teaser_legs  # noqa: E402
from superalgo.weather import game_weather  # noqa: E402

# model weight vs market line (holdout-measured; see RESEARCH_LOG)
W_MARGIN, W_TOTAL = 0.10, 0.05


def _price(x):
    return float(x) if pd.notna(x) and abs(x) >= 100 else -110.0


def offer_bet(sim, home, away, o):
    """Score one sportsbook offer (one side of one market) with the model."""
    side, pt, pr = o["side"], o["point"], o["price"]
    if o["market"] == "spreads":
        if side == home:
            return evaluate_markets(sim, home, away, home_spread=pt, home_spread_price=pr)[0]
        return evaluate_markets(sim, home, away, home_spread=-pt, away_spread_price=pr)[1]
    if o["market"] == "totals":
        return evaluate_markets(sim, home, away, total=pt, over_price=pr, under_price=pr)[0 if side == "Over" else 1]
    if o["market"] == "h2h":
        return evaluate_markets(sim, home, away, home_ml=pr, away_ml=pr)[0 if side == home else 1]
    return None


def power_index(frame, season, week):
    f = frame[(frame["season"] == season) & (frame["week"] == week)]
    rows = []
    for side in ("h", "a"):
        team = f["home_team" if side == "h" else "away_team"]
        rows.append(pd.DataFrame({"team": team.values,
                                  "results_rating": (f[f"{side}_k_points"] - f[f"{side}_k_points"].mean()).values,
                                  "market_rating": (f[f"{side}_k_mkt_points"] - f[f"{side}_k_mkt_points"].mean()).values,
                                  "epa_off": f[f"{side}_k_epa"].values, "pass_epa_off": f[f"{side}_k_pass_epa"].values}))
    t = pd.concat(rows).drop_duplicates("team")
    t["index"] = 0.5 * t["results_rating"] + 0.5 * t["market_rating"]
    t = t.sort_values("index", ascending=False).reset_index(drop=True)
    t["rank"] = t.index + 1
    return [{k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
            for r in t.to_dict("records")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--min-edge", type=float, default=0.03)
    ap.add_argument("--no-weather", action="store_true")
    ap.add_argument("--no-news", action="store_true")
    a = ap.parse_args()

    gp = GamePredictor().build(a.season).fit()
    wk = gp.predict(a.season, a.week)
    sim = GameSimulator()

    # late-breaking QB news (ESPN injury feed): shift projections before the books do
    news = {}
    if not a.no_news:
        try:
            for al in qb_alerts(wk, fetch_injuries(), gp.qbg, a.season, a.week):
                news.setdefault(al["game"], []).append(al)
        except Exception as e:  # noqa: BLE001
            print("news feed unavailable:", e)
    for idx, r in wk.iterrows():
        for al in news.get(f"{r.away_team} @ {r.home_team}", []):
            wk.at[idx, "pred_margin"] += al["home_margin_shift"]
            wk.at[idx, "home_exp"] += al["home_margin_shift"] / 2
            wk.at[idx, "away_exp"] -= al["home_margin_shift"] / 2

    live = None
    if os.environ.get("ODDS_API_KEY"):
        odds = M.odds_to_frame(M.fetch_odds("nfl"))
        live = (odds, M.consensus(odds))

    games = []
    for r in wk.itertuples():
        spread, total = r.spread_line, r.total_line
        cons = None
        if live is not None:
            c = live[1][(live[1].home_team == r.home_team) & (live[1].away_team == r.away_team)]
            if len(c):
                cons = c.iloc[0]
                spread, total = cons.get("mkt_spread", spread), cons.get("mkt_total", total)
        wx = {}
        if not a.no_weather and isinstance(r.stadium_id, str):
            ko = pd.Timestamp(f"{r.gameday} {r.gametime or '13:00'}")
            wx = game_weather(r.stadium_id, ko)
        model_sim = sim.simulate(r.home_exp, r.away_exp, n_sims=a.sims)
        bh, ba = blend_projection(r.home_exp, r.away_exp, spread, total, W_MARGIN, W_TOTAL)
        adj = adjustments(r.home_rest, spread, r.kick_hr, wx.get("wind"), wx.get("roof"),
                          wx.get("gust"), wx.get("precip"))
        m_, t_ = bh - ba + adj["margin"], bh + ba + adj["total"]
        bet_sim = sim.simulate((t_ + m_) / 2, (t_ - m_) / 2, n_sims=a.sims)

        bets = []
        if cons is not None:
            for o in M.offers(live[0], cons["event_id"]):
                b = offer_bet(bet_sim, r.home_team, r.away_team, o)
                if b is not None:
                    bets.append((b, o["book"]))
        elif pd.notna(spread):
            for b in evaluate_markets(
                    bet_sim, r.home_team, r.away_team, home_spread=-spread,
                    home_spread_price=_price(r.home_spread_odds), away_spread_price=_price(r.away_spread_odds),
                    total=total, over_price=_price(r.over_odds), under_price=_price(r.under_odds),
                    home_ml=r.home_moneyline if pd.notna(r.home_moneyline) else None,
                    away_ml=r.away_moneyline if pd.notna(r.away_moneyline) else None):
                bets.append((b, "posted"))
        keep = {id(b) for b in recommend([b for b, _ in bets], a.min_edge)}
        advice = sorted([dict(b.dict(), book=bk) for b, bk in bets if id(b) in keep], key=lambda d: -d["edge"])

        s = model_sim.summary()
        p = s["p_home_win"]
        games.append({
            "game_id": r.game_id, "kickoff": f"{r.gameday} {r.gametime}", "home": r.home_team, "away": r.away_team,
            "home_qb": r.home_qb_name, "away_qb": r.away_qb_name,
            "model": {"home_points": round(s["home_points"], 1), "away_points": round(s["away_points"], 1),
                      "home_win_prob": round(p, 3), "fair_home_spread": s["fair_home_spread"],
                      "fair_total": s["fair_total"], "fair_home_ml": round(prob_to_american(min(max(p, .01), .99))),
                      "fair_away_ml": round(prob_to_american(min(max(1 - p, .01), .99))),
                      "most_likely_scores": [{"home": h, "away": aw, "prob": round(pp, 4)} for h, aw, pp in s["most_likely_scores"]],
                      "qb_edge": {"home": round(float(r.qb_edge_h or 0), 3), "away": round(float(r.qb_edge_a or 0), 3)}},
            "market": {"home_spread": None if pd.isna(spread) else -float(spread),
                       "total": None if pd.isna(total) else float(total),
                       "source": "the-odds-api" if cons is not None else "nflverse posted line"},
            "weather_forecast": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in wx.items()},
            "betting_view": {"home_points": round((t_ + m_) / 2, 1), "away_points": round((t_ - m_) / 2, 1),
                             "home_win_prob": round(bet_sim.p_home_win(), 3), "situational_flags": adj["flags"]},
            "news_alerts": news.get(f"{r.away_team} @ {r.home_team}", []),
            "advice": advice[:3],
            "early_line_signal": line_move_signal(r.home_team, r.away_team, r.pred_margin, spread),
            "teaser_legs": wong_teaser_legs(r.home_team, r.away_team, spread, total),
        })

    report = {"league": "NFL", "season": a.season, "week": a.week, "engine": "v2 (Kalman)",
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "power_index": power_index(gp.frame, a.season, a.week), "games": games}
    out = ROOT / "output" / f"nfl_{a.season}_week{a.week:02d}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=lambda x: None if pd.isna(x) else float(x)))
    print(f"wrote {out}")
    for gm in games:
        m = gm["model"]
        print(f'{gm["away"]:>4} @ {gm["home"]:<4} {m["away_points"]:>5}-{m["home_points"]:<5} P(home)={m["home_win_prob"]:.2f} '
              f'fair {m["fair_home_spread"]:+.1f} mkt {gm["market"]["home_spread"]} | {gm["betting_view"]["situational_flags"]} '
              f'| {[b["selection"] + " " + b["grade"] for b in gm["advice"]] or "-"} | teaser {[l["team"] for l in gm["teaser_legs"]]}'
              f' | early: {(gm["early_line_signal"] or {}).get("bet_now_on", "-")}')


if __name__ == "__main__":
    main()
