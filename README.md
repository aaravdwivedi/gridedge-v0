# GridEdge 🏈

**Confidence-aware fantasy football start/sit recommendations for wide receivers.**

Most fantasy tools give you a single projection and treat it as equally trustworthy every time. GridEdge pairs each projection with a validated risk tier, so you know not just *who* to start, but *how much to trust the call*.

**🔗 Live app:** [gridedge-v0.streamlit.app](https://gridedge-v0.streamlit.app/)

## The Idea

Some player projections are inherently more predictable than others. A player in a stable, high-volume role is easier to project than one with recent volatile scoring. GridEdge tests whether a simple, interpretable confidence signal actually tracks real prediction error — and validates that it does.

## Key Findings

- **Usage beats matchup.** Recent target volume and production account for ~31% of the model's decision-making, vs. ~12% for opponent defensive strength — confirmed independently via SHAP analysis.
- **A model-based confidence signal beats a simple heuristic.** Random Forest tree-variance (how much the model's 400 trees disagree on a prediction) separates accurate from inaccurate predictions **twice as effectively** as a player-volatility-based approach: 105% error gap (2.98 vs. 6.11 MAE) vs. 51% (3.82 vs. 5.78 MAE).
- **Results hold under realistic, time-respecting validation.** Walk-forward testing (train on past weeks, test on the next) produced 4.473 MAE — nearly identical to the original held-out split (4.449).
- **Model choice barely matters here.** Random Forest, Gradient Boosting, Ridge, and Linear Regression all landed within 0.05 MAE of each other, even after adding non-linear-friendly features (target depth, red zone share) — motivating the shift toward confidence-aware output rather than chasing marginal accuracy gains.

Full write-up with methodology and limitations: see `GridEdge_Writeup.md`.

## How It Works

1. **Data:** NFL play-by-play data (2021–2024), filtered to WRs, aggregated to player-game level via `nflreadpy`
2. **Features:** Lagged rolling averages (3/5/8-game targets, receptions, yards, PPR), season averages, target share, target depth (aDOT), red zone share, and opponent defensive strength allowed to the position
3. **Model:** Random Forest Regressor, trained on the above features to predict weekly PPR points, validated via held-out season split, walk-forward testing, and SHAP
4. **Risk tiering:** Each prediction is labeled Low / Medium / High Risk based on Random Forest tree-variance (model-internal uncertainty), validated against actual prediction error

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

- Feature importance may understate correlated rolling-window features (mean decrease in impurity bias) — mitigated by SHAP analysis, which confirms the same conclusion
- Tree-variance may partly track projection magnitude rather than pure uncertainty — not yet isolated; the most important open question for the confidence signal
- Currently WR-only; other positions would need re-validation
- Four-season training window (2021–2024); more seasons would strengthen validation further

## Built By

Aarav Dwivedi
