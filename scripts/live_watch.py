"""Live in-game pricing from ESPN's free scoreboard feed.

    python scripts/live_watch.py --season 2026 --week 3            # one snapshot
    python scripts/live_watch.py --season 2026 --week 3 --loop 20  # poll every 20s

Uses the pre-game projections from output/nfl_<season>_week<NN>.json (run
predict_week.py first) and re-simulates the rest of each live game from its
current score, clock and possession (Markov drive model, superalgo/live.py).
ESPN's own win probability is shown for comparison.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from superalgo.live import live_price  # noqa: E402
from superalgo.news import ESPN_TEAM  # noqa: E402
from superalgo.simulate import GameSimulator  # noqa: E402

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def snapshot(pre: dict, sim: GameSimulator) -> list[dict]:
    with urllib.request.urlopen(SCOREBOARD, timeout=30) as r:
        d = json.loads(r.read())
    out = []
    for e in d.get("events", []):
        st = e["status"]
        if st["type"]["state"] != "in":
            continue
        c = e["competitions"][0]
        teams = {x["homeAway"]: x for x in c["competitors"]}
        home = ESPN_TEAM.get(teams["home"]["team"]["abbreviation"], teams["home"]["team"]["abbreviation"])
        away = ESPN_TEAM.get(teams["away"]["team"]["abbreviation"], teams["away"]["team"]["abbreviation"])
        hs, as_ = int(teams["home"]["score"]), int(teams["away"]["score"])
        period, clock = int(st.get("period", 1)), float(st.get("clock", 0))
        secs = max(0, (4 - period) * 900 + clock) if period <= 4 else clock
        sit = c.get("situation", {})
        poss = sit.get("possession") or sit.get("lastPlay", {}).get("team", {}).get("id")
        home_ball = poss == teams["home"]["team"]["id"]
        g = pre.get(f"{away} @ {home}")
        if g is None:
            continue
        he, ae = g["betting_view"]["home_points"], g["betting_view"]["away_points"]
        ytg = None
        if sit.get("possessionText") and sit.get("yardLine") is not None:
            side_abbr = sit["possessionText"].split()[0]
            poss_abbr = teams["home" if home_ball else "away"]["team"]["abbreviation"]
            y = float(sit["possessionText"].split()[-1])
            ytg = 100 - y if side_abbr == poss_abbr else y
        res = live_price(sim, he, ae, hs, as_, secs, home_ball, n_sims=10000, yardline_100=ytg)
        espn = sit.get("lastPlay", {}).get("probability", {}).get("homeWinPercentage")
        out.append({"game": f"{away} @ {home}", "score": f"{as_}-{hs}", "period": period,
                    "clock": st.get("displayClock"), "home_has_ball": home_ball, "yards_to_goal": ytg,
                    "our_home_win_prob": round(res.p_home_win(), 3),
                    "espn_home_win_prob": espn,
                    "proj_final": f"{res.away.mean():.1f}-{res.home.mean():.1f}",
                    "live_fair_total": float(sorted(res.total)[len(res.total) // 2])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--loop", type=int, default=0, help="seconds between polls (0 = once)")
    a = ap.parse_args()
    pre = json.loads((ROOT / "output" / f"nfl_{a.season}_week{a.week:02d}.json").read_text())
    pre = {f'{g["away"]} @ {g["home"]}': g for g in pre["games"]}
    sim = GameSimulator()
    while True:
        snap = snapshot(pre, sim)
        print(json.dumps(snap, indent=1) if snap else "No games in progress.", flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
