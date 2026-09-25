"""Fit a Kalman rating per statistic and save pre-game predictions.

Hyperparameters are fitted ONLY on 2012-2022, so 2023-2025 stays a clean
holdout. Output: <data>/kalman_preds.parquet (one row per team-game with a
pre-game predicted value of every statistic for that team's offence).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import DATA_DIR, load_games  # noqa: E402
from superalgo.features import canon  # noqa: E402
from superalgo.kalman import fit_params, run_filter  # noqa: E402

FIT_SEASONS = list(range(2012, 2023))

METRICS = {  # stat -> weight column (None = equal weight per game)
    "points": None, "mkt_points": None,
    "epa": "plays", "epa_luck": "plays", "epa_noto": "plays", "success": "plays",
    "explosive": "plays", "pass_epa": "plays", "rush_epa": "plays", "early_epa": "plays",
    "cpoe": "plays", "proe": None, "sack_rate": "plays", "pass_rate": "plays",
    "st_epa": None, "fg_oe_pts": None, "turnovers": None, "fumbles_lost": None,
    "ints": None, "all_plays": None, "drives": None, "rz_td": "rz_trips",
}


def team_game_table() -> pd.DataFrame:
    tg = pd.read_parquet(DATA_DIR / "team_games.parquet")
    g = load_games()
    g = g[g["season"] >= 2010].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(canon)
    g["mkt_home"] = (g["total_line"] + g["spread_line"]) / 2
    g["mkt_away"] = (g["total_line"] - g["spread_line"]) / 2
    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        d = g[["game_id", "season", "week", f"{side}_team", f"{opp_side}_team", f"{side}_score", f"mkt_{side}", "location"]].copy()
        d.columns = ["game_id", "season", "week", "team", "opp", "points", "mkt_points", "location"]
        d["is_home"] = float(side == "home") * (d["location"] != "Neutral")
        rows.append(d.drop(columns="location"))
    base = pd.concat(rows)
    stats = tg.drop(columns=["season", "week", "home_team", "is_home", "game_date", "opp"])
    return base.merge(stats, on=["game_id", "team"], how="left")


def _fit_one(args):
    t, m, w = args
    obs = t.dropna(subset=[m]) if w is None else t.dropna(subset=[m, w])
    p = fit_params(obs, m, w, seasons_fit=FIT_SEASONS)
    r = run_filter(t, m, p, w)  # all rows: unplayed games get predictions too
    print(m, {k: round(float(v), 5) for k, v in p.__dict__.items()}, flush=True)
    return m, p.__dict__, r[["game_id", "team", "pred", "pred_var"]]


def main():
    from multiprocessing import Pool
    t = team_game_table()
    t.to_parquet(DATA_DIR / "team_game_table.parquet")
    preds = t[["game_id", "season", "week", "team", "opp", "is_home"]].copy()
    params = {}
    with Pool(4) as pool:
        for m, prm, r in pool.imap_unordered(_fit_one, [(t, m, w) for m, w in METRICS.items()]):
            params[m] = {k: float(v) for k, v in prm.items()}
            preds = preds.merge(r.rename(columns={"pred": f"k_{m}", "pred_var": f"kv_{m}"}),
                                on=["game_id", "team"], how="left")
    preds.to_parquet(DATA_DIR / "kalman_preds.parquet")
    (DATA_DIR / "kalman_params.json").write_text(json.dumps(params, indent=1))


if __name__ == "__main__":
    main()
