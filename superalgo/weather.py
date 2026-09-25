"""Game-time weather FORECASTS from Open-Meteo (free, no key, non-commercial).

Uses forecasts rather than the observed wind/temperature in the nflverse
schedule, because observed weather isn't known when a bet is placed.
* live / upcoming games: api.open-meteo.com/v1/forecast
* backtests (2022+): historical-forecast-api.open-meteo.com, which stores
  the forecasts that were issued at the time
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

# stadium_id -> (lat, lon, roof)  roof: "outdoors", "dome" (fixed), "retractable"
STADIUMS = {
    "ATL97": (33.755, -84.401, "retractable"), "BAL00": (39.278, -76.623, "outdoors"),
    "BOS00": (42.091, -71.264, "outdoors"), "BUF00": (42.774, -78.787, "outdoors"),
    "BUF01": (42.774, -78.787, "outdoors"), "CAR00": (35.226, -80.853, "outdoors"),
    "CHI98": (41.862, -87.617, "outdoors"), "CIN00": (39.095, -84.516, "outdoors"),
    "CLE00": (41.506, -81.700, "outdoors"), "DAL00": (32.748, -97.093, "retractable"),
    "DEN00": (39.744, -105.020, "outdoors"), "DET00": (42.340, -83.046, "dome"),
    "GNB00": (44.501, -88.062, "outdoors"), "HOU00": (29.685, -95.411, "retractable"),
    "IND00": (39.760, -86.164, "retractable"), "JAX00": (30.324, -81.637, "outdoors"),
    "KAN00": (39.049, -94.484, "outdoors"), "LAX01": (33.953, -118.339, "dome"),
    "LAX99": (34.014, -118.288, "outdoors"), "LAX97": (33.864, -118.261, "outdoors"),
    "MIA00": (25.958, -80.239, "outdoors"), "MIN01": (44.974, -93.258, "dome"),
    "NAS00": (36.166, -86.771, "outdoors"), "NOR00": (29.951, -90.081, "dome"),
    "NYC01": (40.814, -74.074, "outdoors"), "OAK00": (37.752, -122.201, "outdoors"),
    "PHI00": (39.901, -75.168, "outdoors"), "PHO00": (33.528, -112.263, "retractable"),
    "PIT00": (40.447, -80.016, "outdoors"), "SDG00": (32.783, -117.120, "outdoors"),
    "SEA00": (47.595, -122.332, "outdoors"), "SFO01": (37.403, -121.970, "outdoors"),
    "STL00": (38.633, -90.188, "dome"), "TAM00": (27.976, -82.503, "outdoors"),
    "VEG00": (36.091, -115.184, "dome"), "WAS00": (38.908, -76.864, "outdoors"),
    "LON00": (51.556, -0.280, "outdoors"), "LON01": (51.456, -0.342, "outdoors"),
    "LON02": (51.604, -0.066, "outdoors"), "MEX00": (19.303, -99.150, "outdoors"),
    "FRA00": (50.069, 8.645, "outdoors"), "GER00": (48.219, 11.625, "outdoors"),
    "MUN01": (48.219, 11.625, "outdoors"), "SAO00": (-23.545, -46.474, "outdoors"),
    "MAD01": (40.453, -3.688, "retractable"), "DUB00": (53.335, -6.228, "outdoors"),
}

FORECAST = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
VARS = "wind_speed_10m,wind_gusts_10m,precipitation,temperature_2m"


def _get(url: str, params: dict, tries: int = 4) -> dict:
    q = urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            time.sleep(2 ** i)
    return {}


def game_weather(stadium_id: str, kickoff_et: pd.Timestamp, historical: bool = False) -> dict:
    """Average forecast over the ~3.5 hours of the game. ``kickoff_et`` is US
    Eastern time, as in the nflverse schedule."""
    if stadium_id not in STADIUMS:
        return {}
    lat, lon, roof = STADIUMS[stadium_id]
    if roof == "dome":
        return {"roof": roof, "wind": 0.0, "gust": 0.0, "precip": 0.0, "temp": 70.0}
    day = kickoff_et.strftime("%Y-%m-%d")
    d = _get(ARCHIVE if historical else FORECAST, {
        "latitude": lat, "longitude": lon, "start_date": day, "end_date": day, "hourly": VARS,
        "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "timezone": "America/New_York"})
    h = d.get("hourly")
    if not h:
        return {"roof": roof}
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    w = df[(df["time"] >= kickoff_et.floor("h")) & (df["time"] <= kickoff_et + pd.Timedelta(hours=3))]
    return {"roof": roof, "wind": float(w["wind_speed_10m"].mean()), "gust": float(w["wind_gusts_10m"].mean()),
            "precip": float(w["precipitation"].sum()), "temp": float(w["temperature_2m"].mean())}
