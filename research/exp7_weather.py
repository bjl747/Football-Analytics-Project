"""Experiment 7: do archived game-time FORECASTS (not observed weather) beat
the closing total? 2022-2025, outdoor + retractable-roof stadiums."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import DATA_DIR, load_games  # noqa: E402

w = pd.read_parquet(DATA_DIR / "weather_forecasts.parquet")
g = load_games().merge(w, on="game_id")
g = g[g["total"].notna() & g["total_line"].notna()]
# retractable roofs are usually closed in bad weather: judge them by the actual roof call only
# for info; the betting rule applies to true outdoor stadiums
out = g[g["fc_roof"] == "outdoors"].copy()
out["tr"] = out["total"] - out["total_line"]
print("games:", len(out), " corr(forecast wind, observed wind) =",
      round(out[["fc_wind", "wind"]].dropna().corr().iloc[0, 1], 3))
for lo, hi in ((0, 9), (10, 14), (15, 19), (20, 99)):
    d = out[out["fc_wind"].between(lo, hi + 0.999)]
    ok = d["tr"] != 0
    print(f"forecast wind {lo:>2}-{hi:<2} mph: n={len(d):>3}  under {float((d.tr[ok] < 0).mean()):.3f}  "
          f"avg total vs line {d.tr.mean():+.2f}")
for lab, m in (("precip >= 2mm", out["fc_precip"] >= 2), ("temp <= 32F", out["fc_temp"] <= 32),
               ("gusts >= 25mph", out["fc_gust"] >= 25)):
    d = out[m]; ok = d["tr"] != 0
    print(f"{lab:<16} n={len(d):>3} under {float((d.tr[ok] < 0).mean()):.3f} avg {d.tr.mean():+.2f}")
b = np.polyfit(out["fc_wind"].clip(0, 25), out["tr"], 1)
print("slope: points vs closing total per mph of forecast wind:", round(b[0], 3))
