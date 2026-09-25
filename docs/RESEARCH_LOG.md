# Research log: making the engine smarter

Every idea was tested the same way, so we don't fool ourselves:

- **Development:** seasons 2012–2022, leave-one-season-out. The model never sees the season it's predicting.
- **Holdout:** seasons 2023–2025, checked only for finalists. An idea counts only if it works in **both**.
- **Break-even:** a spread bet at standard -110 odds needs a **52.4%** win rate.

Scripts live in `research/` (`exp1`–`exp6`). Outside research came from two research
passes covering nfelo, Open Source Football, FTN/DVOA, ESPN FPI, SP+, Glickman & Stern's
state-space papers, Moskowitz (Journal of Finance 2021), a 2024 academic rest-effect paper,
Action Network, Unexpected Points and Pinnacle/Buchdahl. Reddit itself blocks automated
readers, so community ideas came secondhand.

---

## New building blocks

| Block | What's new vs. version 1 |
|---|---|
| **Kalman-filter ratings** (`superalgo/kalman.py`) | Each team's offence and defence rating in 22 different stats is tracked as a moving target: week-to-week drift plus offseason regression. The filter learns each stat's own stability. For example, field-goal luck got near-zero carry-over, which confirms it's noise. |
| **Luck-stripped stats** (`superalgo/features.py`) | Fumble-luck-adjusted EPA, turnover-free EPA, early-down EPA, pass vs rush EPA, CPOE, pass rate over expected, explosive rate, special-teams EPA, field-goal luck, red-zone TD rate, pace |
| **Market memory** | The same Kalman filter run on past closing lines: what the betting market has believed about each team |
| **QB model with CPOE** (`superalgo/players.py`) | Starter vs. usual-starter value, shrunk toward replacement level |
| **Injury burden** (`superalgo/injuries.py`) | Snap share lost to Out/Doubtful players, by position group, from free injury reports plus snap counts |
| **Context** | Rest and byes, time-zone travel, West Coast teams in early games, divisional games, playoffs |
| **Opening lines** | 2010–2021 from the sportsbookreviewsonline archive and 2024–2025 from ESPN's odds feed (the free sources don't cover 2022–2023). This finally lets us test against lines you could bet early in the week. |

---

## Experiment 1: which stats predict margins? (dev)

| Inputs | Avg miss (pts) | Wins vs closing line when disagreeing by 3+ |
|---|---|---|
| Points ratings only | 10.27 | 51.8% |
| EPA only | 10.35 | 49.0% |
| Luck-adjusted EPA | 10.34 | 50.1% |
| All 21 performance stats | 10.31 | 51.8% |
| **Market memory only** | 10.20 | 53.7% |
| All + market memory | 10.20 | 54.8% |

On their own, the performance stats (EPA and the rest) add almost nothing beyond points ratings for predicting margins. Market memory is the strongest single input.

## Experiment 2: does the market overreact? ❌ Failed holdout

When this week's closing line strayed 3+ points from the market's own smoothed history, betting back toward that history won **56.1%** in 2012–2022 but **47.3%** in 2023–2025. The effect is real in older data but has disappeared recently. **Not used for betting.**

## Experiment 3: beat the opening line / predict line movement ❌ Failed holdout

- The model's disagreement with the opening line predicts **which way the line will move 57–58% of the time**, correlation 0.33. That's genuine information.
- Against the opening line it won **54–64%** in 2012–2021 but **46–52%** in 2024–2025.
- **Conclusion:** the NFL market has become much sharper since about 2023. Keep monitoring it with our own opening-line snapshots (see DATA_SOURCES.md).

## Experiment 4: situational edges (23 rules tested)

| Rule | 2012–2022 | 2023–2025 | Verdict |
|---|---|---|---|
| **Fade home favourites coming off a bye** | 54.8% (155) | 59.4% (32) | ✅ Used (matches a 2024 academic finding that the market overprices byes) |
| **Primetime unders** | 52.8% (593) | 55.1% (187) | ✅ Used, small |
| **Wong teasers** (6-point legs through 3 and 7) | 75.5% per leg | 74.4% per leg | ✅ Flagged. Break-even is ~73.9% at -120; totals ≤49 do better (77.1% / 75.4%) |
| Backup-QB spots, divisional dogs, late-season unders, West Coast early games, home dogs, dome overs… | mixed | failed | ❌ Not used |

We tested 23 rules, so about one could pass by pure luck. The bye and teaser results have independent outside evidence behind them, which is why we trust them more.

## Experiment 5: best pure prediction model

On the **2023–2025 holdout** (855 games):

| Model | Avg miss on margin |
|---|---|
| Version 1 engine | 10.21 |
| Points ratings only (Kalman) | 10.26 |
| **Engine v2: market memory + points + QB + injuries + rest/travel (Ridge)** | **10.04** |
| Vegas closing line | 9.79 |

**Engine v2 cut the gap to Vegas from 0.43 to 0.26 points**, using only information available before each game. Gradient boosting (XGBoost) did no better than the simpler Ridge model.

## Experiment 6: totals

The ratings didn't beat the closing total on their own: 10.28 vs 10.12 average miss on the holdout.

**Wind is the exception.** Using wind measured during the game:
- games with **10–19 mph wind went under 56–58%** in both periods;
- calm games went over 55% in 2023–2025.

That's an upper bound, because bettors only have the forecast. Experiment 7 below tests it with forecasts.

## Experiment 7: forecast weather vs the closing total ✅ Adopted

Archived game-time forecasts from Open-Meteo, 2022–2025, 748 outdoor games. Forecast and observed wind are correlated at 0.70.

| Forecast | Games | Went under | Avg vs closing total |
|---|---|---|---|
| Wind 0–9 mph | 585 | 50.0% | +0.72 |
| **Wind 10–14 mph** | 137 | **60.3%** | **−1.90** |
| Wind 15–19 mph | 23 | 50.0% | −1.24 |
| **Gusts 25+ mph** | 73 | **66.7%** | **−2.83** |
| **Rain 2mm+ during the game** | 31 | **74.2%** | **−6.15** |
| Temp ≤ 32°F | 64 | 51.6% | −1.07 |

Each mph of forecast wind is worth about −0.31 points against the closing total. This agrees with the observed-wind results
in both 2012–22 and 2023–25 (Experiment 6) and with published research (Borghesi 2007; nflanalytic).
The totals market underreacts to wind, gusts and rain. The engine applies shrunk adjustments:
wind 10+ mph −1.0 to −1.3, gusts 25+ −1.0, rain −2.0.

**Caveat:** archived forecasts are short-range (issued close to kickoff), so these edges are for **game-day** betting.

## Experiment 8: more model weight early in the season? ❌ Not stable

The best model weight for weeks 1–4 was 6% in 2012–2022 but 71% in 2023–2025; other week bands also flip.
That's noise, not a pattern, so the engine keeps one fixed weight.

## Experiment 9: predicting line movement ✅ Passed holdout (the strongest finding)

When the pure v2 model disagrees with the **opening** line, the line tends to move toward the model before kickoff.
We ran it two ways, because QB starters and final injury reports aren't known when lines open.

**A. Only information available at the open** (ratings, rest, travel; no same-week news):

| Model − opening line | Dev 2012–21: moves our way | Avg points gained | Holdout 2024–25: moves our way | Avg points gained |
|---|---|---|---|---|
| any | 58.6% | +0.51 | 53.3% | +0.20 |
| ≥ 1.5 pts | 63.2% | +0.82 | 58.8% | +0.46 |
| ≥ 3 pts | 66.7% | +1.43 | 62.2% | +0.48 |

**B. Including same-week QB changes and injury reports** (a "speed" edge: the model reprices the moment news
breaks, and the bettor must act before the books move):

| Model − opening line | Dev: moves our way | Avg points gained | Holdout: moves our way | Avg points gained |
|---|---|---|---|---|
| ≥ 1.5 pts | 69.9% | +1.28 | 69.5% | +1.07 |
| ≥ 3 pts | 76.8% | +2.30 | 79.3% | +1.68 |

Beating the closing number ("closing line value", CLV) is the standard professional measure of skill.
It's far less noisy than win/loss records (Buchdahl). So the engine's edges are **timing** (bet at the open when
it disagrees) and **speed** (reprice instantly on news). By kickoff the market has usually caught up (Experiments 2–3).

**Caveats:** the ESPN feed's "open" time isn't documented, and B depends on how fast news reaches the engine.
Recording our own time-stamped line snapshots is the next validation step.

## Experiment 10: QB value from EPA + CPOE composite ❌ No gain

Blending completion % over expected into QB value (weights 0, 0.3, 0.5) changed holdout error by less than
0.01 points (10.042 / 10.045 / 10.052). EPA alone stays. CPOE is still stored for the player-props project.

## What this means for betting

1. **Against NFL closing spreads, public-data models have no reliable edge in recent seasons.** Our improvements make the *predictions* better, but the closing line already knows what we know. The engine therefore leans on the market (90% market, 10% model) for spreads.
2. **The edges that survived** are structural: bye-week favourites, teasers through key numbers, primetime unders,
   **weather unders (wind, gusts, rain)**, and line shopping across books.
3. **Early lines and fast news are where the model beats the market.** Betting at the open earns about
   +0.5 points of closing line value on the holdout. Reacting instantly to QB and injury news earns +1 to +1.7
   (Experiment 9). Weekly output includes an `early_line_signal`.
4. **College football is likely softer**, per the research (thin Group-of-Five markets). It needs the CFBD key.

## Ideas queued for the next rounds
- A joint Kalman filter that treats the closing line as a second observation (Glickman & Stern style)
- Adaptive market-blend weight by recent model-vs-market error (nfelo-style); the week-of-season version failed (Exp 8)
- Interception and penalty luck adjustments (nfelo WEPA), plus pressure-rate stability from FTN participation data
- Weather forecasts pulled 24–48 hours before kickoff, when lines are still soft
- College: SP+-style preseason prior (returning production 66/19/15 weighting plus recruiting)
