"""Experiment 4: situational / structural edges from the research, each
checked on 2012-2022 (dev) AND 2023-2025 (holdout). A rule only counts if
it clears break-even in both periods.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import DATA_DIR, load_games  # noqa: E402

g = load_games()
g = g[g["season"].between(2012, 2025) & g["result"].notna() & g["spread_line"].notna()].copy()
e2 = pd.read_parquet(DATA_DIR / "exp2_frame.parquet")[["game_id", "qb_edge_h", "qb_edge_a"]]
g = g.merge(e2, on="game_id", how="left")
g["r"] = g["result"] - g["spread_line"]           # home ATS margin
g["tr"] = g["total"] - g["total_line"]            # over margin
PT = {"ARI": -7, "LA": -8, "LAC": -8, "SD": -8, "STL": -6, "SF": -8, "SEA": -8, "LV": -8, "OAK": -8, "DEN": -7}
g["home_tz"] = g["home_team"].map(PT).fillna(-5)
g["away_tz"] = g["away_team"].map(PT).fillna(-5)
g["kick_hr"] = g["gametime"].fillna("13:00").str[:2].astype(int)


def rule(name, mask, side):
    """side: +1 bet home ATS, -1 bet away ATS, 'over'/'under' for totals."""
    out = []
    for lab, per in (("dev", g["season"] <= 2022), ("hold", g["season"] >= 2023)):
        d = g[mask & per]
        v = d["tr"] if side in ("over", "under") else d["r"]
        sgn = 1 if side in (1, "over") else -1
        ok = v != 0
        out.append((lab, round(float((np.sign(v[ok]) == sgn).mean()), 3) if ok.sum() else None, int(ok.sum())))
    flag = "  <-- passes both" if all(o[1] and o[1] > 0.524 and o[2] >= 30 for o in out) else ""
    print(f"{name:<55} {out}{flag}")


rule("home favourite off bye (fade: bet away)", (g.home_rest >= 13) & (g.spread_line > 0), -1)
rule("any team off bye vs rested opp (bet opp) - home bye", (g.home_rest >= 13) & (g.away_rest < 13), -1)
rule("away off bye (fade: bet home)", (g.away_rest >= 13) & (g.home_rest < 13), 1)
rule("home mini-bye (rest 10-11) (bet home)", g.home_rest.between(10, 11) & (g.away_rest <= 7), 1)
rule("away mini-bye (bet away)", g.away_rest.between(10, 11) & (g.home_rest <= 7), -1)
rule("west-coast away team, 1pm ET kickoff (bet home)", (g.away_tz <= -7) & (g.kick_hr <= 13) & (g.home_tz >= -5), 1)
rule("east team at west coast, late game (bet home)", (g.away_tz >= -5) & (g.home_tz <= -7), 1)
rule("home underdogs (bet home)", g.spread_line < 0, 1)
rule("home dogs 3+ (bet home)", g.spread_line <= -3, 1)
rule("big road favourites 7+ (bet home dog)", g.spread_line <= -7, 1)
rule("divisional dogs 3+ (bet dog)", (g.div_game == 1) & (g.spread_line.abs() >= 3), None) if False else None
div_dog = (g.div_game == 1) & (g.spread_line.abs() >= 3)
d_home_dog = div_dog & (g.spread_line < 0)
rule("divisional home dogs 3+ (bet home)", d_home_dog, 1)
rule("divisional away dogs 3+ (bet away)", div_dog & (g.spread_line > 0), -1)
rule("backup QB home favourite 7+ (bet away)", (g.qb_edge_h < -0.05) & (g.spread_line >= 7), -1)
rule("backup QB home team (bet away)", (g.qb_edge_h < -0.08), -1)
rule("backup QB away team (bet home)", (g.qb_edge_a < -0.08), 1)
rule("backup QB away team (bet AWAY, market overreacts?)", (g.qb_edge_a < -0.08), -1)
rule("playoffs home favourite (bet home)", (g.game_type != "REG") & (g.spread_line > 0), 1)
rule("primetime unders", g.kick_hr >= 19, "under")
rule("high totals 49+ under", g.total_line >= 49, "under")
rule("low totals <=38 over", g.total_line <= 38, "over")
rule("divisional unders", g.div_game == 1, "under")
rule("late-season (wk15+) unders", (g.week >= 15) & (g.game_type == "REG"), "under")
rule("dome over", g.roof.isin(["dome", "closed"]), "over")

# Wong teasers: 6-point legs through 3 and 7
print("\nWong teaser legs (need ~72-74% at modern prices):")
for lab, per in (("dev", g["season"] <= 2022), ("hold", g["season"] >= 2023)):
    d = g[per]
    fav_h = d[d.spread_line.between(7.5, 8.5)]; dog_h = d[d.spread_line.between(-2.5, -1.5)]
    fav_a = d[d.spread_line.between(-8.5, -7.5)]; dog_a = d[d.spread_line.between(1.5, 2.5)]
    # home fav teased down 6: home covers (spread-6); away dog teased up 6
    legs = pd.concat([(fav_h.result > fav_h.spread_line - 6), (dog_h.result > dog_h.spread_line - 6),
                      (fav_a.result < fav_a.spread_line + 6), (dog_a.result < dog_a.spread_line + 6)])
    low = pd.concat([fav_h, dog_h, fav_a, dog_a]).total_line <= 49
    print(lab, "all legs", round(legs.mean(), 3), len(legs), "| total<=49", round(legs[low.values].mean(), 3), int(low.sum()))
