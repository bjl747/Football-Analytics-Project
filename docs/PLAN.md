# Project Super-Algo: build plan and status

This maps every piece of the research brief to what is built, what is waiting
on a free API key, and what is deferred (usually because it costs money).
Priority order set by the owner: **game predictions first** (points, most
likely outcomes, betting advice). Player performance comes second.
**Stay free wherever possible.**

## Cost summary

| Item | Cost | Status |
|---|---|---|
| NFL play-by-play, schedules, closing lines (nflverse) | Free, no key | In use |
| College data (CollegeFootballData.com) | Free key, 1,000 calls/month | Adapter written, waiting on key |
| Live sportsbook odds (The Odds API) | Free key, 500 calls/month | Adapter written, waiting on key |
| Compute | Runs on a laptop (backtest takes a few minutes) | Free |
| AWS Next Gen Stats player tracking | Not sold to the public | Not possible. Free NGS summaries are a later option |
| SIS Total Points charting | Paid (thousands per year) | Deferred |
| Kafka/Flink streaming, real-time feed | Paid hosting plus a paid data feed | Deferred. The live engine already runs in plain Python |

## Brief vs. build

### Layer 1: Data and feature engineering
- [x] Historical NFL play-by-play ingestion (`superalgo/data.py`)
- [x] XGBoost Expected Points model, 7 next-score classes, with down, distance,
      field position, clock, roof, home field, timeouts and era features (`epa.py`).
      Matches the public nflverse model (r = 0.99 EP, 0.98 EPA)
- [x] Win Probability GAM (`win_prob.py`). Beats the public nflverse WP on 2023
      (Brier 0.152 vs 0.159)
- [ ] College EP/EPA: use CFBD's free PPA (their EPA) at first (`cfb.py`)
- [ ] Tracking data (NGS): not available to the public; skipped

### Layer 2: Power ratings and adjustments
- [x] Least-squares ratings via the normal equation, with a Bayesian prior and
      an automatically decaying prior weight (`ratings.py`)
- [x] Massey and Colley matrices
- [x] Opponent-adjusted EPA/play with the garbage-time filter (dual-state, state 1)
- [x] Preseason prior from last season, regressed to the mean
- [x] Market-implied ratings (the same maths, solved on past closing lines)
- [x] QB valuation plus starter-change adjustment (`players.py`), the first
      "depth chart moves the spread" feature
- [ ] College prior from returning production plus recruiting (`cfb.preseason_prior`, needs key)

### Layer 3: Probability distributions and pricing
- [x] Dual-state Monte Carlo drive simulator with garbage-time re-injection
      (`simulate.py`), calibrated to real NFL results including key numbers 3 and 7
- [x] Poisson / Negative Binomial / ZIP / ZINB with maximum-likelihood fitting and
      AIC model choice (`distributions.py`). Research finding: team TD counts are
      *under*-dispersed, so the game engine uses the drive model and these are kept
      for player props
- [x] Spread, total, moneyline and exact-score probabilities; fair lines

### Layer 4: Live and bankroll
- [x] Markov-chain live pricing from any game state (`live.py`)
- [x] Fractional Kelly with uncertainty shrink and a stake cap (`odds.py`)
- [x] Betting advice with edge, EV, stake and grade (`advice.py`)
- [x] Multi-book line shopping and sharp consensus (`market.py`, needs key)
- [ ] Kafka/Flink/WebSocket deployment (paid; later)

### Validation
- [x] Walk-forward backtest against real closing lines (`scripts/run_backtest.py`)
- [x] Unit tests (`tests/`)

**Backtest findings (2023-2025, 855 games, out of sample):**
- Pure model margin error 10.2 vs closing line 9.8; totals 10.3 vs 10.1.
- The model's projections are well calibrated (outcome ~ 1.00 x model margin) but
  know less than the closing line, so a regression gives the model about 7% weight
  for spreads and 12% for totals on top of the market. Advice uses this blend.
- Blended spread picks: 31 bets, 17-14, +7% ROI (too small a sample to trust).
  Long-shot moneylines lost money, so they are now filtered out.
- The simulator's win probabilities match market-implied probabilities at every
  spread (e.g. 3-point favourite 59.9% vs market 59.6%).
- Tuning showed small gains from the settings (10.08-10.20 error range). The
  garbage-time cut-off (1%/99% vs 10%/90%) made no measurable difference.

## Next steps, in order
1. **Keys arrive:** run college ingestion and calibration; test the odds feed.
2. **Opening-line tracking:** save Odds API snapshots daily. Closing lines are nearly
   unbeatable. The edge is betting earlier in the week and shopping between books.
   We need our own history of early lines to prove that, and it costs nothing
   but storage.
3. **Wrap the engine as a small web API** (for example FastAPI on a free tier) so
   every app reads the same predictions.
4. **Player project:** extend `players.py` to WR, RB, TE and defence, and use
   `distributions.py` for props.
5. **Refinements:** weather from free sources, travel and rest, coaching
   aggressiveness, current-drive field position in live pricing.
