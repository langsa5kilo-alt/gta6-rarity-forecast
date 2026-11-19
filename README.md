[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/langsa5kilo-alt/gta6-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/langsa5kilo-alt/gta6-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

This is a standard-library Python entry for make.fun's `gta6-trophy-rarity-forecast-01` competition. It forecasts five earn-rate quantiles for every GTA VI trophy.

The model transforms percentages to logits and fits weighted linear trends against log minutes since release over the latest 12 reads. Recent observations receive more weight. Individual slopes are shrunk 30% toward the pooled slope for trophies in the same tier and current-rate tercile. Predictions are constrained near plausible current tier ranges. Forecast uncertainty combines the fitted residual variation with a `0.06 * sqrt(horizon hours)` term, then maps the five normal quantiles back to percentages. Results are rounded to four decimals with strictly increasing quantiles.

Run both included practice cases with:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=. MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` produces a compact JSON file at `out/<id>.out`.
