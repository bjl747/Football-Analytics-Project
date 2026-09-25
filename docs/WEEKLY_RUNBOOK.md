# Weekly runbook (NFL)

What to run and when. Each step is one command. Replace `N` with the week number.

| When | Command | What it does |
|---|---|---|
| **Sunday night / Monday** (lines just opened) | `python scripts/predict_week.py --season 2026 --week N` | Fresh ratings from last week's games. Projections, fair lines, advice, and the **early-line signal**. The early signal is the best time to bet: lines move toward the model about 59% of the time. |
| **Wednesday–Saturday**, a few times a day | `python scripts/news_watch.py --season 2026 --week N` | Checks ESPN injury news. Flags starting QBs ruled Out or Doubtful and how far the spread *should* move. Bet before the books adjust. |
| **Game day morning** | `python scripts/predict_week.py --season 2026 --week N` again | Adds weather forecasts (wind unders), final injury reports, and teaser legs |
| **During games** | `python scripts/live_watch.py --season 2026 --week N --loop 20` | Live win probability and projected final score every 20 seconds, compared with ESPN's |

## With the free API keys
- `export ODDS_API_KEY=...` makes `predict_week.py` compare against every sportsbook. That's line shopping: the advice picks the best price.
- `export CFBD_API_KEY=...` unlocks college football (next build step).

## Reading the output (`output/nfl_<season>_week<NN>.json`)
- **model**: the engine's own projection (score, win %, fair spread/total/moneyline, most likely scores)
- **betting_view**: the engine blended with the market, plus tested adjustments (byes, primetime, wind)
- **advice**: bets with an edge. Grades A/B/C; the stake is a share of bankroll (quarter-Kelly, capped at 3%).
- **early_line_signal**: "bet now on X", with its historical hit rate
- **news_alerts**: QB injury news already applied to the projection
- **teaser_legs**: candidates for 6-point teasers through 3 and 7

*Reminder: even real edges lose often in the short run. Track closing line value (did the line move your way?)
rather than wins and losses, and never bet more than you can afford to lose.*
