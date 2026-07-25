# GridEdge: Confidence-Aware Fantasy Football Start/Sit Recommendations

## The Problem

Most fantasy football projection tools output a single number — "this player will score 14.2 points" — and treat every prediction as equally trustworthy. In practice, some predictions are far more reliable than others. A player with a stable, high-volume role is easier to project accurately than one entering an unpredictable game script. Treating all point estimates as equally confident can lead to worse start/sit decisions than the underlying model quality would suggest.

## Research Question

Can a fantasy football projection model be made more useful not by chasing marginal accuracy gains, but by pairing each prediction with a validated confidence signal — and does that signal actually track real prediction error?

## Methodology

**Data:** NFL play-by-play data (2021–2024 seasons), filtered to wide receivers, aggregated to player-game level using `nflreadpy`.

**Features:** Rolling averages of targets, receptions, yards, and PPR points over the last 3, 5, and 8 games (properly lagged to avoid data leakage — no feature uses information from the game being predicted), season-to-date averages, target share, recent scoring volatility, and opponent defensive strength (fantasy points/targets/yards/TDs allowed to the position over the opponent's last 5 games).

**Models compared:** Baseline (season average), Linear Regression, Ridge Regression, Random Forest, Gradient Boosting.

**Risk labeling:** Two confidence signals were built and compared head-to-head: a volatility-based label (standard deviation of a player's PPR points over their last 3 games) and a model-based label (variance across the Random Forest's 400 individual trees for a given prediction).

**Validation:** In addition to a standard held-out test split (train: 2021–2023, test: 2024), results were re-validated using walk-forward testing — training only on past weeks and testing on the next, rolled forward across every week in the dataset — to confirm findings hold under realistic, time-respecting conditions.

## Key Findings

**1. Usage dominates matchup context.** Feature importance analysis showed recent target volume and PPR production accounted for **~31%** of the Random Forest's total decision-making, while opponent defensive strength features combined accounted for **~12%**. SHAP analysis independently confirmed this: opponent defensive features ranked as the four least-impactful features in the entire model. This suggests that for weekly WR fantasy scoring, *how much a player is used* is a stronger predictor than *who they're playing against* — a quantitative confirmation of a claim fantasy analysts often make qualitatively. Adding two additional non-linear-friendly features (average depth of target, red zone target share) did not change this conclusion or meaningfully improve model performance.

**2. A model-based confidence signal substantially outperforms a simple heuristic.** Both confidence signals were validated by grouping test-set predictions into Low/Medium/High tiers and checking actual prediction error (MAE) within each:

| Signal | Low Tier MAE | High Tier MAE | Error Gap |
|---|---|---|---|
| Volatility-based (player history) | 3.82 | 5.78 | +51% |
| **Tree-variance-based (model uncertainty)** | **2.98** | **6.11** | **+105%** |

The model-based signal — derived from how much the Random Forest's 400 individual trees disagree with each other on a given prediction — separates reliable from unreliable predictions roughly **twice as effectively** as a signal based purely on the player's own scoring history. This was corroborated by a direct residual-correlation test: player volatility correlated with prediction error at 0.186, while tree variance correlated at **0.324** — nearly double.

**3. Model choice barely matters here.** Random Forest, Gradient Boosting, Ridge, and Linear Regression all clustered within 0.05 MAE of each other, with Random Forest improving only marginally over a naive season-average baseline (4.45 vs. 4.58 MAE). This implies the ceiling on point-accuracy for this dataset is close to reached — motivating the shift toward confidence-aware output as the more productive direction, rather than continuing to chase small accuracy gains.

**4. Results hold under realistic, time-respecting validation.** Walk-forward validation (training only on past weeks, testing on the next) produced 4.473 MAE, nearly identical to the original held-out split (4.449) — indicating the reported accuracy is not an artifact of the train/test split and generalizes to realistic week-to-week deployment. A week-by-week breakdown showed no meaningful trend in error across the season (correlation with `week`: -0.005), ruling out seasonal drift as a factor in model reliability.

## The App

These findings are deployed in a live tool ([gridedge-v0.streamlit.app](https://gridedge-v0.streamlit.app/)) where a user compares two wide receivers and sees, side by side: each player's point projection, risk tier, the size of the projected gap between them, and a plain-language explanation of the recommendation (recent usage trend, production trend, and volatility).

## Limitations

- **Feature importance bias:** Random Forest importance (mean decrease in impurity) is biased toward continuous, correlated features. Several rolling-window features (3/5/8-game targets) measure overlapping information, which likely splits and understates their true combined importance.
- **Tree-variance magnitude confound (untested):** tree variance may partly track the magnitude of the point projection itself (higher-scoring players could naturally show more variance across trees) rather than purely reflecting prediction reliability. This has not yet been isolated and is the most important open question for validating the confidence signal further.
- **Data window:** four seasons of training data (2021–2024). A longer window would allow more robust validation, especially for less common risk/matchup combinations, and would better capture year-to-year shifts in offensive scheme and usage.
- **Position scope:** currently WR-only; the framework would need re-validation for other positions with different usage patterns (e.g., RBs, TEs).

## Future Work

- Isolate whether tree variance reflects true model uncertainty independent of projection magnitude
- Incorporate Vegas point totals/spreads as a game-script proxy
- Extend to additional positions
- Expand training window as more seasons become available
- Hyperparameter tuning via GridSearchCV to rule out conservative default settings as a factor in the tied model performance
