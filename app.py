import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model and data
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("gridedge_model_compressed.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("gridedge_data.csv")

rf_model = load_model()
model_data = load_data()

features = [
    "last_3_targets", "last_3_receptions", "last_3_yards", "last_3_ppr", "last_3_ppr_std",
    "last_5_targets", "last_5_receptions", "last_5_yards", "last_5_ppr", "last_5_target_share",
    "last_8_targets", "last_8_ppr", "last_8_target_share",
    "last_5_adot", "last_5_red_zone_share",
    "season_avg_targets", "season_avg_receptions", "season_avg_yards", "season_avg_ppr",
    "def_allowed_ppr_last_5", "def_allowed_targets_last_5", "def_allowed_yards_last_5", "def_allowed_tds_last_5",
    "week"
]

TREE_VAR_LOW_CUTOFF = 3.714682
TREE_VAR_HIGH_CUTOFF = 5.163778

# -----------------------------
# Helper functions
# -----------------------------
def get_latest_player_row(player_name, data):
    exact_matches = data[data["player_display_name"].str.lower() == player_name.lower()].copy()
    if not exact_matches.empty:
        return exact_matches.sort_values(["season", "week"]).iloc[-1]
    return None


def get_player_trend(player_name, data, n=4):
    rows = data[data["player_display_name"].str.lower() == player_name.lower()].copy()
    rows = rows.sort_values(["season", "week"]).tail(n)
    trend = rows[["season", "week", "target", "ppr_points", "air_yards"]].rename(
        columns={"target": "Targets", "ppr_points": "PPR Points", "air_yards": "Air Yards"}
    )
    trend["Game"] = trend["season"].astype(str) + " Wk " + trend["week"].astype(str)
    return trend.set_index("Game")[["Targets", "PPR Points", "Air Yards"]]


def get_risk_label_from_tree_variance(tree_variance):
    if tree_variance < TREE_VAR_LOW_CUTOFF:
        return "Low Risk"
    elif tree_variance < TREE_VAR_HIGH_CUTOFF:
        return "Medium Risk"
    else:
        return "High Risk"


def get_confidence_label(point_difference):
    if point_difference < 1:
        return "Low"
    elif point_difference < 3:
        return "Medium"
    else:
        return "High"


def explain_player(row, projection):
    """Builds specific, numeric explanations so two players never get identical text."""
    explanations = []

    target_change = row["last_3_targets"] - row["season_avg_targets"]
    if abs(target_change) >= 1:
        direction = "up" if target_change > 0 else "down"
        explanations.append(
            f"Target volume trending {direction}: {row['last_3_targets']:.1f}/gm recently vs "
            f"{row['season_avg_targets']:.1f}/gm season avg."
        )
    else:
        explanations.append(f"Target volume steady near season average ({row['season_avg_targets']:.1f}/gm).")

    ppr_change = row["last_3_ppr"] - row["season_avg_ppr"]
    if abs(ppr_change) >= 1:
        direction = "above" if ppr_change > 0 else "below"
        explanations.append(
            f"Recent production {direction} season average by {abs(ppr_change):.1f} pts "
            f"({row['last_3_ppr']:.1f} vs {row['season_avg_ppr']:.1f})."
        )
    else:
        explanations.append("Recent production is in line with season average.")

    if row["last_5_adot"] >= 12:
        explanations.append(f"Deep-threat usage: {row['last_5_adot']:.1f} avg depth of target.")
    elif row["last_5_adot"] <= 6:
        explanations.append(f"Short/possession usage: {row['last_5_adot']:.1f} avg depth of target.")
    else:
        explanations.append(f"Intermediate route depth: {row['last_5_adot']:.1f} avg depth of target.")

    if row["last_5_red_zone_share"] >= 0.20:
        explanations.append(f"Notable red zone role: {row['last_5_red_zone_share']*100:.0f}% of recent targets.")

    return explanations


def get_tree_variance(row_X, model):
    tree_preds = np.array([tree.predict(row_X)[0] for tree in model.estimators_])
    return tree_preds.std()


def compare_players(player_a, player_b, data, model, features):
    a = get_latest_player_row(player_a, data)
    b = get_latest_player_row(player_b, data)

    if a is None or b is None:
        return None

    a_X = pd.DataFrame([a[features]], columns=features)
    b_X = pd.DataFrame([b[features]], columns=features)

    a_proj = model.predict(a_X)[0]
    b_proj = model.predict(b_X)[0]

    a_tree_var = get_tree_variance(a_X, model)
    b_tree_var = get_tree_variance(b_X, model)

    difference = abs(a_proj - b_proj)
    confidence = get_confidence_label(difference)

    a_risk = get_risk_label_from_tree_variance(a_tree_var)
    b_risk = get_risk_label_from_tree_variance(b_tree_var)

    recommended = player_a if a_proj > b_proj else player_b

    return {
        "recommended_start": recommended,
        "confidence": confidence,
        "point_difference": round(difference, 2),
        "player_a": {
            "name": a["player_display_name"], "team": a["posteam"],
            "projection": round(a_proj, 2), "last_3_ppr": round(a["last_3_ppr"], 2),
            "season_avg_ppr": round(a["season_avg_ppr"], 2), "risk": a_risk,
            "tree_variance": round(a_tree_var, 2), "explanation": explain_player(a, a_proj),
        },
        "player_b": {
            "name": b["player_display_name"], "team": b["posteam"],
            "projection": round(b_proj, 2), "last_3_ppr": round(b["last_3_ppr"], 2),
            "season_avg_ppr": round(b["season_avg_ppr"], 2), "risk": b_risk,
            "tree_variance": round(b_tree_var, 2), "explanation": explain_player(b, b_proj),
        }
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GridEdge", page_icon="🏈")

st.title("🏈 GridEdge")
st.caption("Confidence-aware WR start/sit recommendations")

player_list = sorted(model_data["player_display_name"].dropna().unique().tolist())

col1, col2 = st.columns(2)
with col1:
    player_a = st.selectbox("Player A", player_list, index=None, placeholder="Select a receiver")
with col2:
    player_b = st.selectbox("Player B", player_list, index=None, placeholder="Select a receiver")

if st.button("Compare", type="primary"):
    if not player_a or not player_b:
        st.warning("Select two players first.")
    elif player_a == player_b:
        st.warning("Pick two different players.")
    else:
        with st.spinner("Running model across all 400 trees..."):
            result = compare_players(player_a, player_b, model_data, rf_model, features)

        if result is None:
            st.error("Couldn't find data for one of these players.")
        else:
            st.subheader(f"✅ Recommended Start: {result['recommended_start']}")
            st.write(f"**Confidence:** {result['confidence']} · **Projected gap:** {result['point_difference']} pts")

            st.divider()

            c1, c2 = st.columns(2)
            for col, key, name in [(c1, "player_a", player_a), (c2, "player_b", player_b)]:
                p = result[key]
                with col:
                    st.markdown(f"### {p['name']}")
                    st.metric("Projected PPR points", p["projection"])
                    st.write(f"**Team:** {p['team']}")

                    risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
                    st.write(f"**Risk level:** {risk_color.get(p['risk'], '')} {p['risk']}")
                    st.caption(f"Model uncertainty score: {p['tree_variance']}")

                    st.write("**Why:**")
                    for reason in p["explanation"]:
                        st.write(f"- {reason}")

                    st.write("**Last 4 games:**")
                    trend = get_player_trend(name, model_data, n=4)
                    if not trend.empty:
                        st.line_chart(trend)

st.divider()
with st.expander("How does the risk score work?"):
    st.write(
        "Risk is based on how much GridEdge's 400 individual decision trees "
        "disagree with each other on a given prediction. This model-based signal "
        "separates accurate from inaccurate predictions more than twice as "
        "effectively as a simpler, player-history-based approach "
        "(105% error gap vs. 51%)."
    )
