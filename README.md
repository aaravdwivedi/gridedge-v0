# gridedge-v0
# GridEdge v0.1

GridEdge is a fantasy football analytics project focused on building confidence-aware start/sit recommendations. Most fantasy tools give a single point projection, but this project explores whether a model can also estimate how trustworthy that projection is.

This early version builds a weekly wide receiver fantasy football projection model using historical NFL play-by-play data. It engineers player usage, recent performance, volatility, and matchup-context features, then compares multiple models against a simple baseline.

## Current Status

This is an early prototype, not a finished product. The current notebook focuses on building the first working WR projection model and testing whether machine learning can improve over simple fantasy-point baselines.

## Features So Far

* Loads NFL play-by-play data from multiple seasons
* Creates weekly WR fantasy football data
* Engineers rolling usage and recent performance features
* Adds matchup-context features based on opponent defensive performance
* Compares baseline, linear regression, ridge regression, random forest, and gradient boosting models
* Includes an early start/sit comparison function
* Begins exploring player risk using volatility/error patterns

## Research Question

Can matchup-context features and model uncertainty estimates improve weekly fantasy football start/sit recommendations by identifying when player point projections are more or less trustworthy?

## Next Steps

* Add true Random Forest tree-spread uncertainty
* Validate whether higher uncertainty corresponds to higher real prediction error
* Create confidence buckets for start/sit decisions
* Build a Streamlit app for player comparisons
* Improve matchup context with Vegas lines, pace, and game-script features
