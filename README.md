# Project Super-Algo: football prediction and odds engine

The engine behind the apps. It predicts NFL games (college is next): projected
points, win chances, most likely final scores, fair betting lines, and betting
advice with bet sizes. It runs entirely on **free data**.

See [`docs/PLAN.md`](docs/PLAN.md) for the full build plan, what's done, and costs.

## What it does, in plain terms

1. **Scores every play** with our own machine-learning Expected Points model.
2. **Tracks every team as a moving target.** A Kalman filter (the maths behind GPS
   tracking) follows each team's offence and defence across 22 stats, week by week.
   It learns how "sticky" each stat really is. Fumble luck and field-goal luck
   turn out to be almost pure noise; passing efficiency is real.
3. **Remembers what the betting market believed** about each team in past weeks.
4. **Adjusts for the people on the field**: quarterback changes (EPA plus completion
   percentage over expected), injured starters by position group, rest and byes,
   and cross-country travel.
5. **Plays each game 20,000 times**, drive by drive, calibrated so margins land on 3
   and 7 as often as real NFL games do.
6. **Gives betting advice** only where it has held up out of sample: sportsbook lines
   are the anchor, plus the situational edges that survived testing (bye-week
   favourites, primetime and wind unders, teasers through 3 and 7), sized with quarter-Kelly.

## Honest results (2023–2025 holdout, 855 games the model never trained on)

| | Average miss on final margin |
|---|---|
| Version 1 engine | 10.21 pts |
| **Engine v2 (current)** | **10.04 pts** |
| Vegas closing line | 9.79 pts |

- Engine v2 cut the gap to Vegas from 0.43 to 0.26 points.
- Against *closing* spreads, no public-data model we tested had a reliable edge in
  2023–2025. The market has gotten much sharper. So the engine leans on the market
  line for spreads and takes only the edges that passed both test periods.
- Full experiment write-ups: [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).
- Where to get live weekly data (mostly free): [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
- What to run each day of the week: [`docs/WEEKLY_RUNBOOK.md`](docs/WEEKLY_RUNBOOK.md).

## Edges that survived testing (out of sample)

| Edge | Evidence |
|---|---|
| **Bet early when the model disagrees with the opening line** | Line moves our way 59% of the time (+0.5 pts of closing line value) in 2024–25 |
| **React to QB injury news before the books** (`news_watch.py`) | Lines move our way 70–79% of the time (+1.1 to +1.7 pts) when news is in the model first |
| **Weather unders** (forecast wind 10+ mph, gusts 25+, rain) | 60–74% unders in 2022–25 |
| **Fade home favourites off a bye** | 54.8% (2012–22), 59.4% (2023–25) |
| **Wong teasers through 3 and 7** | 75.5% / 74.4% per leg (needs ~73.9%) |
| **Primetime unders** | 52.8% / 55.1% |

## Sample output

[`output/nfl_2026_week03.json`](output/nfl_2026_week03.json) shows real predictions
for NFL Week 3, 2026: a ranked power index plus, for every game, projected score,
win probability, fair spread/total/moneyline, top 5 most likely scores, and advice.
This file is what the apps read.

## API keys (both free)

| Key | Where | Unlocks |
|---|---|---|
| `ODDS_API_KEY` | https://the-odds-api.com (500 calls/month free) | Live odds from many sportsbooks, line shopping |
| `CFBD_API_KEY` | https://collegefootballdata.com/key | College football |

## For developers

```bash
pip install -r requirements.txt
python -m pytest tests                                  # unit tests
python scripts/predict_week.py --season 2026 --week 3   # weekly predictions, engine v2 (~2 min)
python scripts/run_backtest.py                          # v1 walk-forward backtest (~7 min)
python research/build_ratings.py && python research/exp5_full_model.py   # v2 research pipeline
```

Data downloads automatically into `data/` (set `SUPERALGO_DATA` to change that).

| Module | Role |
|---|---|
| `superalgo/data.py` | Free nflverse data loaders |
| `superalgo/epa.py` | XGBoost Expected Points (7 next-score outcomes) and EPA |
| `superalgo/win_prob.py` | Win-probability GAM (splines plus logistic) |
| `superalgo/ratings.py` | Normal-equation least squares, Bayesian priors, Massey, Colley, opponent-adjusted EPA |
| `superalgo/players.py` | QB value and starter-change adjustment |
| `superalgo/calibrate.py` | Measures league constants (drives, scoring rates, late-game behaviour) |
| `superalgo/simulate.py` | Dual-state Monte Carlo drive simulator (Markov chain) with key-number calibration |
| `superalgo/distributions.py` | Poisson, Negative Binomial, ZIP and ZINB fitting (for player props) |
| `superalgo/live.py` | In-game re-pricing from any score, clock and possession |
| `superalgo/odds.py` | Odds maths, vig removal, Fractional Kelly |
| `superalgo/advice.py` | Market evaluation and bet recommendations |
| `superalgo/market.py` | The Odds API client, sharp consensus, line shopping |
| `superalgo/cfb.py` | College adapter (CFBD). Written, untested until the key arrives |
| `superalgo/pipeline.py` | **Engine v2**: Kalman ratings, context, margin and total models |
| `superalgo/kalman.py` | Dynamic opponent-adjusted Kalman-filter ratings |
| `superalgo/features.py` | Team-game stats, including luck-stripped versions |
| `superalgo/injuries.py` | Injury burden from injury reports plus snap counts |
| `superalgo/situational.py` | Tested situational adjustments and teaser legs |
| `superalgo/weather.py` | Game-time weather forecasts (Open-Meteo) |
| `superalgo/engine.py` | Engine v1 (kept for comparison) and the market blend |
| `research/` | The experiment lab: every test in the research log |
| `superalgo/backtest.py` | Walk-forward evaluation against closing lines |

*Responsible use: this is a probabilistic model. Even real edges lose often in the short run. Never bet more than you can afford to lose.*
