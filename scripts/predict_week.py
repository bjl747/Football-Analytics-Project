"""Produce the weekly NFL prediction file the apps read.

    python scripts/predict_week.py --season 2026 --week 3

Output: output/nfl_<season>_week<NN>.json containing
  * power_index : every team's rating (points better/worse than average)
  * games       : projected score, win probability, fair spread/total/moneyline,
                  most likely final scores, and betting advice

Market prices come from The Odds API when ODDS_API_KEY is set (live, many
books). Otherwise the posted line and moneyline in the free nflverse schedule
file are used.
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
from superalgo.data import load_games, load_pbp  # noqa: E402
from superalgo.engine import GameEngine, blend_projection, build_efficiency, prepare_games  # noqa: E402
from superalgo.epa import EPModel  # noqa: E402
from superalgo.odds import prob_to_american  # noqa: E402
from superalgo.players import qb_game_table  # noqa: E402
from superalgo.win_prob import WinProbModel  # noqa: E402


def build(season: int, week: int, history: int = 6):
    games = prepare_games(load_games())
    seasons = list(range(season - history, season + 1))
    model_train = [s for s in seasons[:-1]][-5:]
    train = load_pbp(model_train)
    ep, wp = EPModel().fit(train), WinProbModel().fit(train)
    pbps = {s: load_pbp(s) for s in seasons}
    eff = pd.concat([build_efficiency(p, ep, wp) for p in pbps.values()])
    qbg = pd.concat([qb_game_table(p) for p in pbps.values()])
    engine = GameEngine(qb_games=qbg)
    stack_seasons = seasons[1:-1]
    feats = engine.walk_forward_features(games, eff, stack_seasons)
    engine.fit_stacker(feats, games)
    return engine, games, eff


def power_index(engine, games, eff, season, week) -> list[dict]:
    pts, er = engine.ratings_asof(games, eff, season, week)
    mkt = engine.market_ratings_asof(games, season, week)
    t = pts.table().set_index("team")
    t["market_net"] = pd.Series(mkt.net) if mkt else np.nan
    t = t.join(er, how="left")
    # Super-Algo Index: blend of results-based, efficiency-based and market views,
    # expressed as points better than an average team on a neutral field
    epa_pts = t["adj_net"].fillna(0) * 62  # ~62 offensive plays per game
    t["index"] = 0.4 * t["net"] + 0.3 * epa_pts + 0.3 * t["market_net"].fillna(t["net"])
    t = t.sort_values("index", ascending=False)
    t["rank"] = range(1, len(t) + 1)
    return [{"rank": int(r["rank"]), "team": team, "index": round(r["index"], 2),
             "points_rating": round(r["net"], 2), "offense": round(r["offense"], 2),
             "defense": round(r["defense"], 2), "market_rating": round(float(r["market_net"]), 2),
             "adj_epa_off": round(float(r["adj_off"]), 3), "adj_epa_def": round(float(r["adj_def"]), 3)}
            for team, r in t.iterrows()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--min-edge", type=float, default=0.03)
    a = ap.parse_args()

    engine, games, eff = build(a.season, a.week)
    wk = games[(games["season"] == a.season) & (games["week"] == a.week)]
    pts, er = engine.ratings_asof(games, eff, a.season, a.week)
    mkt = engine.market_ratings_asof(games, a.season, a.week)
    feats = engine.side_features(wk, pts, er, a.season, a.week, mkt=mkt)
    proj = engine.game_projections(feats)

    live = None
    if os.environ.get("ODDS_API_KEY"):
        odds = M.odds_to_frame(M.fetch_odds("nfl"))
        live = (odds, M.consensus(odds))

    out_games = []
    for gid, r in proj.iterrows():
        g = wk.set_index("game_id").loc[gid]
        spread, total = g["spread_line"], g["total_line"]
        cons = None
        if live is not None:
            c = live[1]
            c = c[(c.home_team == r.home_team) & (c.away_team == r.away_team)]
            if len(c):
                cons = c.iloc[0]
                spread, total = cons.get("mkt_spread", spread), cons.get("mkt_total", total)
        model_sim = engine.simulate(r.home_exp, r.away_exp, a.sims)
        bh, ba = blend_projection(r.home_exp, r.away_exp, spread, total, engine.cfg.w_margin, engine.cfg.w_total)
        bet_sim = engine.simulate(bh, ba, a.sims)

        bets = []
        if cons is not None:
            for o in M.offers(live[0], cons["event_id"]):
                kw = {}
                if o["market"] == "spreads" and o["side"] == r.home_team:
                    kw = dict(home_spread=o["point"], home_spread_price=o["price"], away_spread_price=-10000)
                elif o["market"] == "totals" and o["side"] == "Over":
                    kw = dict(total=o["point"], over_price=o["price"], under_price=-10000)
                elif o["market"] == "h2h" and o["side"] == r.home_team:
                    kw = dict(home_ml=o["price"], away_ml=-10000)
                if kw:
                    bets += [dict(b.dict(), book=o["book"]) for b in evaluate_markets(bet_sim, r.home_team, r.away_team, **kw)
                             if b.price != -10000]
        elif not np.isnan(spread):
            price = lambda x: float(x) if pd.notna(x) and abs(x) >= 100 else -110.0  # noqa: E731
            bets = [dict(b.dict(), book="posted") for b in evaluate_markets(
                bet_sim, r.home_team, r.away_team, home_spread=-spread,
                home_spread_price=price(g.get("home_spread_odds")), away_spread_price=price(g.get("away_spread_odds")),
                total=total, over_price=price(g.get("over_odds")), under_price=price(g.get("under_odds")),
                home_ml=g.get("home_moneyline") if pd.notna(g.get("home_moneyline")) else None,
                away_ml=g.get("away_moneyline") if pd.notna(g.get("away_moneyline")) else None)]
        books = [b.pop("book") for b in bets]
        objs = [Bet(**b) for b in bets]
        keep = {id(b) for b in recommend(objs, a.min_edge)}
        rec = sorted([dict(b.dict(), book=bk) for b, bk in zip(objs, books) if id(b) in keep], key=lambda b: -b["edge"])

        s = model_sim.summary()
        p_home = s["p_home_win"]
        out_games.append({
            "game_id": gid, "kickoff": str(g["gameday"])[:10], "home": r.home_team, "away": r.away_team,
            "home_qb": g.get("home_qb_name"), "away_qb": g.get("away_qb_name"),
            "model": {"home_points": round(s["home_points"], 1), "away_points": round(s["away_points"], 1),
                      "home_win_prob": round(p_home, 3),
                      "fair_home_spread": s["fair_home_spread"], "fair_total": s["fair_total"],
                      "fair_home_ml": round(prob_to_american(min(max(p_home, 0.01), 0.99))),
                      "fair_away_ml": round(prob_to_american(min(max(1 - p_home, 0.01), 0.99))),
                      "most_likely_scores": [{"home": h, "away": aw, "prob": round(p, 4)}
                                             for h, aw, p in s["most_likely_scores"]]},
            "market": {"home_spread": None if pd.isna(spread) else -float(spread),
                       "total": None if pd.isna(total) else float(total),
                       "source": "the-odds-api" if cons is not None else "nflverse posted line"},
            "betting_view": {"home_points": round(bh, 1), "away_points": round(ba, 1),
                             "home_win_prob": round(bet_sim.p_home_win(), 3)},
            "advice": rec[:3],
        })

    report = {"league": "NFL", "season": a.season, "week": a.week,
              "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "notes": "model = pure Super-Algo projection; betting_view = model blended with market "
                       f"(model weight {engine.cfg.w_margin:.0%} spread / {engine.cfg.w_total:.0%} total), "
                       "which is what the advice is priced from.",
              "power_index": power_index(engine, games, eff, a.season, a.week), "games": out_games}
    out = ROOT / "output" / f"nfl_{a.season}_week{a.week:02d}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=lambda x: None if pd.isna(x) else float(x)))
    print(f"wrote {out}")
    for gm in out_games:
        m = gm["model"]
        print(f'{gm["away"]:>4} @ {gm["home"]:<4} model {m["away_points"]:>5}-{m["home_points"]:<5} '
              f'P(home)={m["home_win_prob"]:.2f} fair spread {m["fair_home_spread"]:+.1f} | market {gm["market"]["home_spread"]} '
              f'| advice: {[b["selection"] + " " + b["grade"] for b in gm["advice"]] or "none"}')


if __name__ == "__main__":
    main()
