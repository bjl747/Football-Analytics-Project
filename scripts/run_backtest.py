"""Honest walk-forward backtest against real NFL closing lines.

* EP and WP models are trained on 2017-2019 only.
* Every weekly prediction uses only games played before that week.
* The stacker and the model-vs-market blend weights are learned on 2020-2022.
* 2023-2025 are the untouched test seasons.

Usage:  python scripts/run_backtest.py
Writes: <data dir>/backtest_summary.json and backtest_ledger.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superalgo.backtest import evaluate, simulate_games  # noqa: E402
from superalgo.data import DATA_DIR, load_games, load_pbp  # noqa: E402
from superalgo.engine import (GameEngine, blend_projection, build_efficiency,  # noqa: E402
                              fit_market_blend, prepare_games)
from superalgo.epa import EPModel  # noqa: E402
from superalgo.players import qb_game_table  # noqa: E402
from superalgo.win_prob import WinProbModel  # noqa: E402

TRAIN_MODEL = [2017, 2018, 2019]
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
STACK_TRAIN = [2020, 2021, 2022]
TEST = [2023, 2024, 2025]


def load_inputs(wp_bounds=(0.01, 0.99)):
    games = prepare_games(load_games())
    eff_path = DATA_DIR / f"eff_{wp_bounds[0]}_{wp_bounds[1]}_{SEASONS[-1]}.parquet"
    qb_path = DATA_DIR / f"qb_games_{SEASONS[-1]}.parquet"
    if not eff_path.exists():
        train = load_pbp(TRAIN_MODEL)
        ep, wp = EPModel().fit(train), WinProbModel().fit(train)
        pd.concat([build_efficiency(load_pbp(s), ep, wp, wp_bounds) for s in SEASONS]).to_parquet(eff_path)
    if not qb_path.exists():
        pd.concat([qb_game_table(load_pbp(s)) for s in TRAIN_MODEL + SEASONS[1:]]).to_parquet(qb_path)
    return games, pd.read_parquet(eff_path), pd.read_parquet(qb_path)


def main():
    games, eff, qbg = load_inputs()
    engine = GameEngine(qb_games=qbg)
    feats = engine.walk_forward_features(games, eff, STACK_TRAIN + TEST)

    # blend weights from leave-one-season-out predictions on the training seasons
    cv = []
    for s in STACK_TRAIN:
        tr = feats[feats["season"].isin(STACK_TRAIN) & (feats["season"] != s)]
        engine.fit_stacker(tr, games)
        cv.append(engine.game_projections(feats[feats["season"] == s]))
    cv = pd.concat(cv).join(games.set_index("game_id")[["result", "total", "spread_line", "total_line"]]).dropna()
    w_m = fit_market_blend(cv.home_exp - cv.away_exp, cv.spread_line, cv.result)
    w_t = fit_market_blend(cv.home_exp + cv.away_exp, cv.total_line, cv.total)

    engine.fit_stacker(feats[feats["season"].isin(STACK_TRAIN)], games)
    proj = engine.game_projections(feats[feats["season"].isin(TEST)])
    g = games.set_index("game_id")

    report = {"blend_weights": {"margin": w_m, "total": w_t},
              "stacker": dict(zip(["intercept"] + list(engine.stacker.feature_names_in_),
                                  [float(engine.stacker.intercept_)] + [float(c) for c in engine.stacker.coef_]))}
    ledgers = []
    for name, p in (("pure_model", proj), ("market_blend", _blend(proj, g, w_m, w_t))):
        sims = simulate_games(engine, p)
        res = evaluate(engine, p, games, sims)
        report[name] = res["summary"]
        by_season = {}
        for s, fr in res["frame"].groupby("season"):
            by_season[int(s)] = {"margin_mae_model": float(((fr.home_exp - fr.away_exp) - fr.result).abs().mean()),
                                 "margin_mae_market": float((fr.spread_line - fr.result).abs().mean())}
        report[name]["by_season"] = by_season
        ledgers.append(res["ledger"].assign(variant=name))
    print(json.dumps(report, indent=2))
    (DATA_DIR / "backtest_summary.json").write_text(json.dumps(report, indent=2))
    pd.concat(ledgers).to_csv(DATA_DIR / "backtest_ledger.csv", index=False)


def _blend(proj, g, w_m, w_t):
    out = proj.copy()
    for gid, r in proj.iterrows():
        h, a = blend_projection(r.home_exp, r.away_exp, g.at[gid, "spread_line"], g.at[gid, "total_line"], w_m, w_t)
        out.at[gid, "home_exp"], out.at[gid, "away_exp"] = h, a
    return out


if __name__ == "__main__":
    main()
