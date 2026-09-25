"""Injury burden: how much regular playing time a team is missing.

For each team and game: find players listed Out or Doubtful on the final
injury report who were *regulars* (average snap share over the team's
previous 4 games), and add up their snap share by position group. A starting
cornerback who plays 100% of snaps and is ruled out adds 1.0 to "DB".
The model learns what each group is worth in points. QBs are handled by
players.py instead.

Free data: nflverse injuries + snap_counts releases (2012 onward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = {
    "QB": "QB", "RB": "SKILL", "FB": "SKILL", "WR": "SKILL", "TE": "SKILL",
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}
STATUS_OUT = {"Out": 1.0, "Doubtful": 0.8}


def _norm(name: pd.Series) -> pd.Series:
    return (name.fillna("").str.lower().str.replace(r"[^a-z ]", "", regex=True)
            .str.replace(r"\s+(jr|sr|ii|iii|iv|v)$", "", regex=True).str.strip())


def injury_burden(injuries: pd.DataFrame, snaps: pd.DataFrame, canon=lambda t: t,
                  lookback: int = 4, regular: float = 0.35) -> pd.DataFrame:
    s = snaps[snaps["game_type"] == "REG"].copy() if "game_type" in snaps else snaps.copy()
    s["team"] = s["team"].map(canon)
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["key"] = _norm(s["player"])
    s = s.sort_values(["season", "week"])
    # rolling average share over the team's previous `lookback` games, per player
    s["prev_share"] = s.groupby(["season", "team", "key"])["share"].transform(
        lambda x: x.shift(1).rolling(lookback, min_periods=1).mean())
    last = s.groupby(["season", "team", "key"]).agg(
        weeks=("week", list), shares=("share", list))

    inj = injuries[injuries["report_status"].isin(STATUS_OUT)].copy()
    inj["team"] = inj["team"].map(canon)
    inj["key"] = _norm(inj["full_name"])
    inj["w_out"] = inj["report_status"].map(STATUS_OUT)
    inj["group"] = inj["position"].map(GROUPS).fillna("OTHER")

    vals = []
    for r in inj.itertuples():
        try:
            wk, sh = last.loc[(r.season, r.team, r.key)]
        except KeyError:
            vals.append(0.0); continue
        prior = [v for w, v in zip(wk, sh) if w < r.week][-lookback:]
        vals.append(float(np.mean(prior)) if prior else 0.0)
    inj["share"] = vals
    inj = inj[inj["share"] >= regular]
    inj["burden"] = inj["share"] * inj["w_out"]
    out = inj.pivot_table(index=["season", "week", "team"], columns="group", values="burden",
                          aggfunc="sum", fill_value=0.0)
    out.columns = [f"inj_{c}" for c in out.columns]
    return out.reset_index()
