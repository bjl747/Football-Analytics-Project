"""Archived game-time weather forecasts for 2022-2025 outdoor/retractable NFL games.

One Open-Meteo request per stadium per season (hourly data for the whole
season), then each game takes the average over its ~3.5 hours.
nflverse gametime is US Eastern, so we request timezone=America/New_York.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superalgo.data import DATA_DIR, load_games  # noqa: E402
from superalgo.weather import ARCHIVE, STADIUMS, VARS, _get  # noqa: E402

g = load_games()
g = g[g["season"].between(2022, 2025) & g["result"].notna()]
g = g[g["stadium_id"].map(lambda s: STADIUMS.get(s, (0, 0, "dome"))[2] != "dome")]
g["ko"] = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"))

rows = []
for (sid, season), gg in g.groupby(["stadium_id", "season"]):
    lat, lon, roof = STADIUMS[sid]
    d = _get(ARCHIVE, {"latitude": lat, "longitude": lon,
                       "start_date": gg["ko"].min().strftime("%Y-%m-%d"),
                       "end_date": gg["ko"].max().strftime("%Y-%m-%d"), "hourly": VARS,
                       "wind_speed_unit": "mph", "temperature_unit": "fahrenheit",
                       "timezone": "America/New_York"}, tries=3)
    time.sleep(0.3)
    print(sid, season, bool(d.get("hourly")), flush=True)
    if not d.get("hourly"):
        print("no data", sid, season, flush=True)
        continue
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    for r in gg.itertuples():
        w = h[(h["time"] >= r.ko.floor("h")) & (h["time"] <= r.ko + pd.Timedelta(hours=3))]
        rows.append({"game_id": r.game_id, "fc_roof": roof, "fc_wind": w["wind_speed_10m"].mean(),
                     "fc_gust": w["wind_gusts_10m"].mean(), "fc_precip": w["precipitation"].sum(),
                     "fc_temp": w["temperature_2m"].mean()})
w = pd.DataFrame(rows)
w.to_parquet(DATA_DIR / "weather_forecasts.parquet")
print(len(w))
print(w.describe())
