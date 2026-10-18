import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "football_model.joblib"
st.set_page_config(page_title="Premier League Match Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Premier League Match Predictor")
st.write("Predict home win, draw, or away win from historical team form.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Put football_model.joblib in the same folder as app.py.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
teams = bundle["teams"]
model = bundle["model"]
numeric_features = bundle["numeric_features"]
team_states = bundle["team_states"]

st.caption(f"Model: {bundle['model_name']} · Historical data through {bundle['trained_through']}")
st.warning("This is a historical-data demonstration, not a live fixture feed. Predictions may be stale unless you retrain using recent seasons.")

home_team = st.selectbox("Home team", teams, index=0)
away_options = [t for t in teams if t != home_team]
away_team = st.selectbox("Away team", away_options, index=0)

def build_features(home, away):
    h = team_states[home]
    a = team_states[away]
    row = {"HomeTeam": home, "AwayTeam": away}
    row.update({f"home_{k}": v for k, v in h.items()})
    row.update({f"away_{k}": v for k, v in a.items()})
    return pd.DataFrame([row])

if st.button("Predict match outcome", type="primary"):
    if home_team == away_team:
        st.error("Choose two different teams.")
        st.stop()
    X = build_features(home_team, away_team)
    probabilities = model.predict_proba(X)[0]
    classes = model.named_steps["model"].classes_
    prob_map = dict(zip(classes, probabilities))
    names = {"H": "Home win", "D": "Draw", "A": "Away win"}
    predicted = model.predict(X)[0]
    st.subheader(f"Prediction: {names.get(predicted, predicted)}")
    cols = st.columns(3)
    for col, label in zip(cols, ["H", "D", "A"]):
        col.metric(names[label], f"{prob_map.get(label, 0.0):.1%}")
    st.write("Probability distribution")
    st.bar_chart(pd.DataFrame({
        "Outcome": [names[k] for k in ["H", "D", "A"]],
        "Probability": [prob_map.get(k, 0.0) for k in ["H", "D", "A"]]
    }).set_index("Outcome"))
    st.caption("Probabilities are model estimates, not guarantees.")
