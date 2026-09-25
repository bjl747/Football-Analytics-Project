import numpy as np
import pandas as pd
import pytest

from superalgo.kalman import KalmanParams, run_filter
from superalgo.situational import adjustments, wong_teaser_legs


def _league(seed=0, n_teams=8, seasons=3, weeks=10):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    true = {t: rng.normal(0, 5) for t in teams}
    rows = []
    for s in range(seasons):
        for w in range(1, weeks + 1):
            order = rng.permutation(teams)
            for gi, (h, a) in enumerate(zip(order[::2], order[1::2])):
                gid = f"{s}_{w}_{gi}"
                for team, opp, home in ((h, a, 1.0), (a, h, 0.0)):
                    rows.append({"game_id": gid, "season": 2000 + s, "week": w, "team": team, "opp": opp,
                                 "is_home": home, "pts": 21 + true[team] + 1.0 * home + rng.normal(0, 3)})
    return pd.DataFrame(rows), true


def test_kalman_learns_team_strength_walk_forward():
    df, true = _league()
    p = KalmanParams(q_week=0.01, q_season=0.5, carry=0.9, r=9.0, prior_var=25.0)
    out = run_filter(df, "pts", p)
    last = out.merge(df[["game_id", "team", "season", "week"]], on=["game_id", "team"])
    last = last[(last.season == 2002) & (last.week == 10)].set_index("team")["off"]
    tr = pd.Series(true)[last.index]
    assert np.corrcoef(tr, last)[0, 1] > 0.9
    # first game of all has no information: prediction equals the prior mean
    assert out["off"].iloc[0] == pytest.approx(0.0)


def test_unplayed_games_get_predictions():
    df, _ = _league(seasons=1, weeks=3)
    df.loc[df.week == 3, "pts"] = np.nan
    out = run_filter(df, "pts", KalmanParams(0.01, 0.5, 0.9, 9.0, 25.0))
    assert out["pred"].notna().all()


def test_situational_rules():
    a = adjustments(home_rest=14, spread_line=6.5, kick_hour=20, forecast_wind=16, roof="outdoors")
    assert a["margin"] < 0 and a["total"] < 0
    assert set(a["flags"]) >= {"home_fav_off_bye", "primetime"}
    assert adjustments(7, -3, 13, 20, "dome")["total"] == 0


def test_wong_legs():
    assert wong_teaser_legs("H", "A", 8.0, 44)[0]["teased_line"] == -2.0
    assert wong_teaser_legs("H", "A", -2.0, 44)[0]["teased_line"] == 8.0
    assert wong_teaser_legs("H", "A", 2.5, 51)[0] == {"team": "A", "teased_line": 8.5, "game": "A @ H",
                                                       "low_total": False, "hist_leg_win": 0.755}
    assert wong_teaser_legs("H", "A", 4.0, 44) == []
