"""Data loading from the free, public nflverse project.

Play-by-play: https://github.com/nflverse/nflverse-data (one parquet per season)
Schedules/lines: https://github.com/nflverse/nfldata (games.csv, includes closing
spread, total and moneylines, which we use as the "market" in backtests).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

DATA_DIR = Path(os.environ.get("SUPERALGO_DATA", Path(__file__).resolve().parent.parent / "data"))


def _cached(url: str, name: str, refresh: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    if refresh or not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def load_pbp(seasons, refresh: bool = False) -> pd.DataFrame:
    """Load nflverse play-by-play for one or more seasons."""
    if isinstance(seasons, int):
        seasons = [seasons]
    frames = [
        pd.read_parquet(_cached(PBP_URL.format(season=s), f"pbp_{s}.parquet", refresh))
        for s in seasons
    ]
    return pd.concat(frames, ignore_index=True)


def load_games(refresh: bool = False) -> pd.DataFrame:
    """Load every NFL game since 1999 with final scores and closing betting lines.

    Note: nflverse ``spread_line`` is from the HOME team's view as a positive
    number when home is favoured (i.e. expected home margin).
    """
    games = pd.read_csv(_cached(GAMES_URL, "games.csv", refresh))
    return games
