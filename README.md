# GridEdge 🏈

**Confidence-aware fantasy football start/sit recommendations for wide receivers.**

Most fantasy tools give you a single projection and treat it as equally trustworthy every time. GridEdge pairs each projection with a validated risk tier, so you know not just *who* to start, but *how much to trust the call*.

**🔗 Live app:** [gridedge-v0.streamlit.app](https://gridedge-v0.streamlit.app/)

## The Idea

Some player projections are inherently more predictable than others. A player in a stable, high-volume role is easier to project than one with recent volatile scoring. GridEdge tests whether a simple, interpretable confidence signal actually tracks real prediction error — and validates that it does.

## Key Findings

- **Usage beats matchup.** Recent target volume and production account for ~31% of the model's decision-making, vs. ~12% for opponent defensive strength — a quantitative confirmation that "target share is king."
- **The risk label is real, not decorative.** Predictions flagged High Risk have 51% higher average error than Low Risk predictions (5.78 vs. 3.82 MAE), validated on held-out test data.
- **Model choice barely matters here.** Random Forest, Gradient Boosting, Ridge, and Linear Regression all landed within 0.05 MAE of each other — motivating the shift toward confidence-aware output rather than chasing marginal accuracy gains.

Full write-up with methodology and limitations: see `GridEdge_Writeup.md`.

## How It Works

1. **Data:** NFL play-by-play data (2021–2024), filtered to WRs, aggregated to player-game level via `nflreadpy`
2. **Features:** Lagged rolling averages (3/5/8-game targets, receptions, yards, PPR), season averages, target share, scoring volatility, and opponent defensive strength allowed to the position
3. **Model:** Random Forest Regressor, trained on the above features to predict weekly PPR points
4. **Risk tiering:** Each prediction is labeled Low / Medium / High Risk based on the player's recent scoring volatility, validated against actual model error

## Repo Contents

| File | Purpose |
|---|---|
| `GridEdge_Model.ipynb` | Full data pipeline, feature engineering, model training, and validation |
| `app.py` | Streamlit app — loads the trained model and serves player comparisons |
| `gridedge_model_compressed.pkl` | Trained Random Forest model |
| `gridedge_data.csv` | Processed player-game data used by the app |
| `requirements.txt` | Python dependencies for deployment |
| `GridEdge_Writeup.md` | Research write-up: question, methodology, findings, limitations |

## Limitations & Next Steps

- Feature importance may understate correlated rolling-window features (mean decrease in impurity bias)
- Risk label is based on player volatility, not the model's internal prediction uncertainty (e.g., Random Forest tree-variance) — a stronger, model-based confidence signal is a natural next step
- Currently WR-only; other positions would need re-validation
- Two-season training window; more seasons would strengthen validation

## Built By

Aarav Dwivedi
