# GridEdge: Confidence-Aware Fantasy Football Start/Sit Recommendations

## The Problem

Most fantasy football projection tools output a single number — "this player will score 14.2 points" — and treat every prediction as equally trustworthy. In practice, some predictions are far more reliable than others. A player with a stable, high-volume role is easier to project accurately than one entering an unpredictable game script. Treating all point estimates as equally confident can lead to worse start/sit decisions than the underlying model quality would suggest.

## Research Question

Can a fantasy football projection model be made more useful not by chasing marginal accuracy gains, but by pairing each prediction with a validated confidence signal — and does that signal actually track real prediction error?

## Methodology

**Data:** NFL play-by-play data (2021–2024 seasons), filtered to wide receivers, aggregated to player-game level using `nflreadpy`.

**Features:** Rolling averages of targets, receptions, yards, and PPR points over the last 3, 5, and 8 games (properly lagged to avoid data leakage — no feature uses information from the game being predicted), season-to-date averages, target share, recent scoring volatility, and opponent defensive strength (fantasy points/targets/yards/TDs allowed to the position over the opponent's last 5 games).

**Models compared:** Baseline (season average), Linear Regression, Ridge Regression, Random Forest, Gradient Boosting.

**Risk labeling:** Each prediction is tagged Low / Medium / High Risk based on the player's recent scoring volatility (standard deviation of PPR points over their last 3 games).

## Key Findings

**1. Usage dominates matchup context.** Feature importance analysis showed recent target volume and PPR production (`last_8_targets`, `last_8_ppr`, `last_8_target_share`) accounted for **~31%** of the Random Forest's total decision-making, while opponent defensive strength features combined accounted for **~12%**. This suggests that for weekly WR fantasy scoring, *how much a player is used* is a stronger predictor than *who they're playing against* — a quantitative confirmation of a claim fantasy analysts often make qualitatively.

**2. The risk label is empirically validated, not decorative.** Grouping test-set predictions by risk tier showed a clean, monotonic relationship between labeled risk and actual prediction error:

| Risk Tier | Avg. Prediction Error (MAE) | Games |
|---|---|---|
| Low Risk | 3.82 | 1,782 |
| Medium Risk | 5.00 | 1,113 |
| High Risk | 5.78 | 523 |

High Risk predictions carry **51% higher average error** than Low Risk predictions. This confirms that a simple, interpretable volatility-based signal can meaningfully flag which projections deserve less trust — the central premise of the project.

**3. Model choice barely matters here.** Random Forest, Gradient Boosting, Ridge, and Linear Regression all clustered within 0.05 MAE of each other, with Random Forest improving only marginally over a naive season-average baseline (4.46 vs. 4.58 MAE). This implies the ceiling on point-accuracy for this dataset is close to reached — motivating the shift toward confidence-aware output as the more productive direction, rather than continuing to chase small accuracy gains.

## The App

These findings are deployed in a live tool ([gridedge-v0.streamlit.app](https://gridedge-v0.streamlit.app/)) where a user compares two wide receivers and sees, side by side: each player's point projection, risk tier, the size of the projected gap between them, and a plain-language explanation of the recommendation (recent usage trend, production trend, and volatility).

## Limitations

- **Feature importance bias:** Random Forest importance (mean decrease in impurity) is biased toward continuous, correlated features. Several rolling-window features (3/5/8-game targets) measure overlapping information, which likely splits and understates their true combined importance.
- **Risk signal is volatility-based, not model-based:** the risk label reflects a player's own historical scoring volatility, not the Random Forest's internal prediction uncertainty (e.g., variance across individual trees). These are related but distinct signals — the volatility-based version is validated here; a model-uncertainty version remains untested.
- **Data window:** four seasons of training data (2021–2024). A longer window would allow more robust validation, especially for less common risk/matchup combinations, and would better capture year-to-year shifts in offensive scheme and usage.
- **Position scope:** currently WR-only; the framework would need re-validation for other positions with different usage patterns (e.g., RBs, TEs).

## Future Work

- Test whether Random Forest tree-variance produces a stronger confidence signal than volatility-based risk labeling
- Incorporate Vegas point totals/spreads as a game-script proxy
- Extend to additional positions
- Expand training window as more seasons become available
