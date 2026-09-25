# Project Super-Algo: football prediction and odds engine

The engine behind the apps. It predicts NFL games (college is next): projected
points, win chances, most likely final scores, fair betting lines, and betting
advice with bet sizes. It runs entirely on **free data**.

See [`docs/PLAN.md`](docs/PLAN.md) for the full build plan, what's done, and costs.

## What it does, in plain terms

1. **Scores every play** (Expected Points Added). A machine-learning model learns
   how many points a team can expect from any situation, such as 3rd and 5 at
   midfield with 2 minutes left. Every play is graded by how much it helped or hurt.
2. **Rates every team** by combining three views: points scored and allowed,
   play-by-play efficiency (with garbage time removed), and what the betting market
   has said about them in past weeks. The maths adjusts for strength of schedule
   and starts each season from a regressed version of last season's rating.
3. **Adjusts for quarterbacks.** If a backup starts, the projection moves.
4. **Plays each game 20,000 times** drive by drive, including how teams really
   behave late in blowouts, which creates "backdoor covers." It's tuned so final
   margins land on 3 and 7 as often as they do in real NFL games.
5. **Compares against the sportsbooks** and suggests only bets where our chance
   of winning is clearly better than the price, sized with quarter-Kelly
   (a cautious, bankroll-protecting formula).

## Honest results (tested on 855 real games, 2023-2025, never seen in training)

| | Our model alone | Vegas closing line |
|---|---|---|
| Average miss on final margin | 10.2 pts | 9.8 pts |
| Average miss on total points | 10.3 pts | 10.1 pts |

- On its own, the model is close to Vegas but not better. That's normal:
  closing lines are the sharpest numbers in sports betting.
- Betting advice therefore uses a **blend** (about 93% market, 7% model for spreads;
  88/12 for totals). The weights were learned from 2020-2022 data. In the test
  seasons, blended spread picks went 17-14 (+7% return), which is too few bets to prove anything.
- Moneyline bets on big underdogs lost money, so the engine no longer recommends them.
- **Where the real edge likely is (next step):** betting early in the week before
  lines sharpen, and shopping prices across many sportsbooks. That needs the free
  odds API key, and we need to collect our own history of early lines to prove it.

Full numbers: [`output/backtest_summary.json`](output/backtest_summary.json).

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
python scripts/run_backtest.py                          # walk-forward backtest (~7 min)
python scripts/predict_week.py --season 2026 --week 3   # weekly predictions (~1 min)
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
| `superalgo/engine.py` | Ties it together: ratings, projection, simulation, market blend |
| `superalgo/backtest.py` | Walk-forward evaluation against closing lines |

*Responsible use: this is a probabilistic model. Even real edges lose often in the short run. Never bet more than you can afford to lose.*
