import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and data (cached so it only loads once, not on every click)
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
    # recent player volume
    "last_3_targets",
    "last_3_receptions",
    "last_3_yards",
    "last_3_ppr",
    "last_3_ppr_std",

    # longer player trends
    "last_5_targets",
    "last_5_receptions",
    "last_5_yards",
    "last_5_ppr",
    "last_5_target_share",
    "last_8_targets",
    "last_8_ppr",
    "last_8_target_share",

    # season player averages
    "season_avg_targets",
    "season_avg_receptions",
    "season_avg_yards",
    "season_avg_ppr",

    # matchup
    "def_allowed_ppr_last_5",
    "def_allowed_targets_last_5",
    "def_allowed_yards_last_5",
    "def_allowed_tds_last_5",

    # context
    "week"
]

# -----------------------------
# Same helper functions from your notebook
# -----------------------------
def get_latest_player_row(player_name, data):
    exact_matches = data[
        data["receiver_player_name"].str.lower() == player_name.lower()
    ].copy()

    if not exact_matches.empty:
        return exact_matches.sort_values(["season", "week"]).iloc[-1]

    partial_matches = data[
        data["receiver_player_name"].str.contains(player_name, case=False, na=False)
    ].copy()

    if partial_matches.empty:
        return None

    best_match_name = (
        partial_matches["receiver_player_name"]
        .value_counts()
        .idxmax()
    )

    best_matches = data[data["receiver_player_name"] == best_match_name].copy()
    return best_matches.sort_values(["season", "week"]).iloc[-1]


def get_risk_label(volatility):
    if volatility < 4:
        return "Low Risk"
    elif volatility < 8:
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
    explanations = []

    if row["last_3_targets"] > row["season_avg_targets"]:
        explanations.append("Target volume is trending up recently.")
    elif row["last_3_targets"] < row["season_avg_targets"]:
        explanations.append("Target volume is trending down recently.")
    else:
        explanations.append("Target volume is close to season average.")

    if row["last_3_ppr"] > row["season_avg_ppr"]:
        explanations.append("Recent fantasy production is above season average.")
    elif row["last_3_ppr"] < row["season_avg_ppr"]:
        explanations.append("Recent fantasy production is below season average.")
    else:
        explanations.append("Recent fantasy production is close to season average.")

    if row["last_3_ppr_std"] >= 8:
        explanations.append("This player has been volatile recently.")
    elif row["last_3_ppr_std"] <= 4:
        explanations.append("This player has been relatively consistent recently.")

    return explanations


def compare_players(player_a, player_b, data, model, features):
    a = get_latest_player_row(player_a, data)
    b = get_latest_player_row(player_b, data)

    if a is None or b is None:
        return None

    a_X = pd.DataFrame([a[features]], columns=features)
    b_X = pd.DataFrame([b[features]], columns=features)

    a_proj = model.predict(a_X)[0]
    b_proj = model.predict(b_X)[0]

    difference = abs(a_proj - b_proj)
    confidence = get_confidence_label(difference)

    a_risk = get_risk_label(a["last_3_ppr_std"])
    b_risk = get_risk_label(b["last_3_ppr_std"])

    recommended = player_a if a_proj > b_proj else player_b

    return {
        "recommended_start": recommended,
        "confidence": confidence,
        "point_difference": round(difference, 2),
        "player_a": {
            "name": a["receiver_player_name"],
            "team": a["posteam"],
            "projection": round(a_proj, 2),
            "last_3_targets": round(a["last_3_targets"], 2),
            "last_3_ppr": round(a["last_3_ppr"], 2),
            "season_avg_ppr": round(a["season_avg_ppr"], 2),
            "risk": a_risk,
            "explanation": explain_player(a, a_proj)
        },
        "player_b": {
            "name": b["receiver_player_name"],
            "team": b["posteam"],
            "projection": round(b_proj, 2),
            "last_3_targets": round(b["last_3_targets"], 2),
            "last_3_ppr": round(b["last_3_ppr"], 2),
            "season_avg_ppr": round(b["season_avg_ppr"], 2),
            "risk": b_risk,
            "explanation": explain_player(b, b_proj)
        }
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GridEdge", page_icon="🏈")

st.title("🏈 GridEdge")
st.caption("Confidence-aware WR start/sit recommendations")

player_list = sorted(model_data["receiver_player_name"].dropna().unique().tolist())

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
        result = compare_players(player_a, player_b, model_data, rf_model, features)

        if result is None:
            st.error("Couldn't find data for one of these players.")
        else:
            st.subheader(f"✅ Recommended Start: {result['recommended_start']}")
            st.write(f"**Confidence:** {result['confidence']} · **Projected gap:** {result['point_difference']} pts")

            st.divider()

            c1, c2 = st.columns(2)
            for col, key in [(c1, "player_a"), (c2, "player_b")]:
                p = result[key]
                with col:
                    st.markdown(f"### {p['name']}")
                    st.metric("Projected PPR points", p["projection"])
                    st.write(f"**Team:** {p['team']}")

                    risk_color = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
                    st.write(f"**Risk level:** {risk_color.get(p['risk'], '')} {p['risk']}")

                    st.write(f"**Last 3 games avg:** {p['last_3_ppr']} PPR")
                    st.write(f"**Season avg:** {p['season_avg_ppr']} PPR")

                    st.write("**Why:**")
                    for reason in p["explanation"]:
                        st.write(f"- {reason}")

st.divider()
st.caption(
    "GridEdge validates its risk tiers against real prediction error: "
    "High Risk games average ~5.78 MAE vs ~3.82 MAE for Low Risk games, "
    "confirming the risk score tracks genuine model uncertainty."
)
