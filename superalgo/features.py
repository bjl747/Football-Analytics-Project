"""Team-game statistics: one row per team per game, from the offence's view.

A team's *defensive* numbers are simply its opponents' offensive rows, so
every metric here can be rated for offence and defence by the same maths.

Beyond plain EPA/play this builds the "luck-stripped" and component signals
the research points to as more predictive of future results:

* pass vs rush EPA (passing is the more stable signal)
* early-down EPA (1st/2nd down, less noisy than 3rd-down outcomes)
* success rate and explosive-play rate
* turnover-neutral EPA (turnovers are largely random from game to game)
* fumble-luck-adjusted EPA (who recovers a fumble is close to a coin flip)
* field-goal luck (makes over expectation given kick distance)
* special teams EPA, pace, and neutral-situation pass rate over expected
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def canon(team):
    return FRANCHISE.get(team, team)


def _fg_expectation(pbp: pd.DataFrame) -> tuple[float, float]:
    """Logistic fit of FG make probability on distance (league-wide)."""
    fg = pbp[pbp["field_goal_attempt"] == 1].dropna(subset=["kick_distance"])
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression().fit(fg[["kick_distance"]], fg["field_goal_result"].eq("made"))
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def team_game_stats(pbp: pd.DataFrame, wp_bounds=(0.02, 0.98), fg_coef=None) -> pd.DataFrame:
    p = pbp[pbp["posteam"].notna()].copy()
    p["posteam"] = p["posteam"].map(canon)
    p["defteam"] = p["defteam"].map(canon)
    scrim = p[p["play_type"].isin(["pass", "run"]) & p["epa"].notna()].copy()
    wp = scrim["wp"].fillna(0.5)
    scrim["live"] = (wp >= wp_bounds[0]) & (wp <= wp_bounds[1])
    scrim["is_pass"] = scrim["pass"].fillna(0).eq(1) if "pass" in scrim else scrim["play_type"].eq("pass")
    scrim["turnover"] = scrim["interception"].fillna(0).eq(1) | scrim["fumble_lost"].fillna(0).eq(1)
    scrim["explosive"] = np.where(scrim["is_pass"], scrim["yards_gained"] >= 20, scrim["yards_gained"] >= 12)
    scrim["early"] = scrim["down"].isin([1, 2])
    # fumble luck: every fumble gets the average value of a fumble, lost or not
    fum = scrim["fumble"].fillna(0).eq(1)
    mean_fum_epa = scrim.loc[fum, "epa"].mean() if fum.any() else 0.0
    scrim["epa_luck"] = np.where(fum, mean_fum_epa, scrim["epa"])
    scrim["neutral"] = scrim["live"] & scrim["down"].isin([1, 2]) & scrim["wp"].between(0.2, 0.8) \
        & (scrim["half_seconds_remaining"] > 120)

    L = scrim[scrim["live"]]
    key = ["game_id", "posteam", "defteam"]
    g = L.groupby(key)
    out = pd.DataFrame({
        "plays": g.size(),
        "epa": g["epa"].mean(),
        "epa_luck": g["epa_luck"].mean(),
        "success": g["success"].mean(),
        "explosive": g["explosive"].mean(),
    })
    out["epa_noto"] = L[~L["turnover"]].groupby(key)["epa"].mean()
    out["pass_epa"] = L[L["is_pass"]].groupby(key)["epa"].mean()
    out["rush_epa"] = L[~L["is_pass"]].groupby(key)["epa"].mean()
    out["early_epa"] = L[L["early"]].groupby(key)["epa"].mean()
    out["pass_rate"] = L.groupby(key)["is_pass"].mean()
    out["proe"] = scrim[scrim["neutral"]].groupby(key)["pass_oe"].mean() / 100.0
    out["cpoe"] = L[L["cpoe"].notna()].groupby(key)["cpoe"].mean() / 100.0
    out["sack_rate"] = L[L["is_pass"]].groupby(key)["sack"].mean()
    # whole-game counts (garbage time included)
    a = scrim.groupby(key)
    out["all_plays"] = a.size()
    out["turnovers"] = a["turnover"].sum()
    out["fumbles"] = a["fumble"].sum()
    out["fumbles_lost"] = a["fumble_lost"].sum()
    out["ints"] = a["interception"].sum()

    # special teams: EPA on kicking plays, credited to the team with the ball
    st = p[p["special_teams_play"].fillna(0).eq(1) & p["epa"].notna()]
    out["st_epa"] = st.groupby(key)["epa"].sum()

    # field-goal luck: points above what kick distances would predict
    fg = p[p["field_goal_attempt"].fillna(0).eq(1) & p["kick_distance"].notna()]
    b0, b1 = fg_coef if fg_coef else _fg_expectation(p)
    fg = fg.assign(exp=1 / (1 + np.exp(-(b0 + b1 * fg["kick_distance"]))),
                   made=fg["field_goal_result"].eq("made"))
    out["fg_oe_pts"] = (3 * (fg["made"] - fg["exp"])).groupby([fg[k] for k in key]).sum()

    # drives and red zone
    d = p[p["fixed_drive"].notna()].groupby(["game_id", "posteam", "defteam", "fixed_drive"]).agg(
        res=("fixed_drive_result", "first"), min_yl=("yardline_100", "min"))
    d["rz"] = d["min_yl"] <= 20
    d["td"] = d["res"].eq("Touchdown")
    dg = d.groupby(level=[0, 1, 2])
    out["drives"] = dg.size()
    out["rz_trips"] = dg["rz"].sum()
    out["rz_td"] = d[d["rz"]].groupby(level=[0, 1, 2])["td"].mean()

    out = out.reset_index().rename(columns={"posteam": "team", "defteam": "opp"})
    meta = p.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                    home_team=("home_team", "first"), game_date=("game_date", "first"))
    meta["home_team"] = meta["home_team"].map(canon)
    out = out.join(meta, on="game_id")
    out["is_home"] = (out["team"] == out["home_team"]).astype(float)
    fill0 = ["st_epa", "fg_oe_pts", "rz_trips"]
    out[fill0] = out[fill0].fillna(0.0)
    return out
