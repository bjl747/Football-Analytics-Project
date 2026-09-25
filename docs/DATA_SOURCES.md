# Where the data comes from, for weekly and live operation

Verified September 2026. ✅ = tested live from our environment during NFL Week 3.

## The free stack (what the engine uses)

| Need | NFL | College |
|---|---|---|
| Play-by-play, schedules, closing lines, rest, QBs | **nflverse** ✅ (free, no key) | **CFBD** (free key, 1,000 calls/month) plus cfbfastR release files ✅ |
| Injuries | nflverse `injuries` ✅ (daily 07:00 UTC); **ESPN injuries** ✅ (near real-time) | ESPN; SEC/ACC availability reports (since 2025) |
| Snap counts / depth charts | nflverse `snap_counts`, `depth_charts` ✅ | CFBD `/player/usage` |
| Opening lines (history) | sportsbookreviewsonline.com archive, 2010–2021 ✅; ESPN odds feed, 2024+ ✅ | SBRO NCAAF archive; CFBD `/lines` (`spreadOpen`) |
| Current lines | ESPN scoreboard odds (DraftKings open and current) ✅; **The Odds API** (free, 500 credits/month) | same, with `americanfootball_ncaaf` |
| Prediction-market prices | **Kalshi** ✅ and **Polymarket** ✅ (free, real-time, no key) | same |
| Live game state | **ESPN scoreboard / summary / plays** ✅ (5–30 seconds behind broadcast) | ESPN `college-football`, `groups=80` for FBS |
| Weather | **Open-Meteo** ✅ (forecasts, plus archived forecasts for backtests); NWS api.weather.gov ✅ | same, using CFBD `/venues` coordinates |
| Advanced stats | nflverse Next Gen Stats, FTN charting, PFR advanced stats | CFBD SP+, FPI, PPA, havoc stats |

## Exact endpoints

**Monday–Tuesday (last week's results)**
- `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet`. Re-pull on Thursday, after stat corrections.
- `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv` (refreshed every 5 minutes in season)
- `.../snap_counts/snap_counts_2026.parquet` and `.../injuries/injuries_2026.parquet`

**Tuesday–Saturday (lines)**
- ESPN: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=2026&seasontype=2&week=N`. Read `competitions[0].odds[0]`, which holds the open and current line.
- ESPN odds per game: `https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{id}/competitions/{id}/odds`
- The Odds API: `/v4/sports/americanfootball_nfl/odds?regions=us,eu&markets=spreads,totals,h2h`. The EU region includes Pinnacle.
- Kalshi: `https://external-api.kalshi.com/trade-api/v2/markets?series_ticker=KXNFLGAME&status=open`
- Polymarket: `https://gamma-api.polymarket.com/events?tag_slug=nfl&closed=false`

**Wednesday–Sunday (injuries and news)**
- ESPN: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries`
- Sleeper: `https://api.sleeper.app/v1/players/nfl` (pull once a day) and `/v1/players/nfl/trending/add`
- Official game status is filed by 4 PM ET two days before kickoff. Inactives come out 90 minutes before kickoff (nfl.com/inactives).

**Game day (weather)**
- `https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..&hourly=wind_speed_10m,wind_gusts_10m,precipitation,temperature_2m&wind_speed_unit=mph&temperature_unit=fahrenheit`

**Live (poll every 10–20 seconds)**
- `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`. The `situation` object has down, distance, yard line, possession and ESPN's win probability.
- `https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={id}` (drives, pickcenter odds)
- `https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{id}/competitions/{id}/plays?limit=300`
- Live prices: Kalshi and Polymarket WebSockets (free), or The Odds API.

## The cheap upgrade stack (about $45/month), when it's worth it

| Item | Cost | Why |
|---|---|---|
| The Odds API 20K plan | $30/month | Live odds from many books, plus **historical 5-minute line snapshots since 2020**. This is the key to measuring closing line value and building opening-line history. |
| CFBD Tier 2 | $5/month | Live college play-by-play, weather and adjusted metrics |
| PFF+ | $10/month | Player grades (manual CSV) for the player project |

## Not available or not worth it
- **Pinnacle API:** closed to the public since July 2025. Get Pinnacle prices through The Odds API's EU region instead.
- **NFL Next Gen Stats tracking:** raw tracking data is not public. Big Data Bowl samples on Kaggle are for research only.
- **Sportradar, Genius Sports, OddsJam API, Unabated API:** enterprise pricing, from hundreds to thousands of dollars a month.
- **Betfair:** not available to US residents.
- **Unofficial DraftKings/FanDuel scraping:** blocked, fragile, and against their terms.

## Terms-of-use notes
- The ESPN and Action Network endpoints are unofficial. They're fine for personal modelling and risky for a commercial product.
- Open-Meteo's free tier is non-commercial.
- nflverse (CC-BY) and CFBD require attribution.
- If the apps become a paid product, budget for licensed feeds: The Odds API, MySportsFeeds, or SportsDataIO.
