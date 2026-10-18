from pathlib import Path
from collections import defaultdict, deque
import glob, json, joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.dummy import DummyClassifier

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# 1) Load every CSV in data/. Each file should contain HomeTeam, AwayTeam,
#    FTHG, FTAG, FTR and Date (common football-data.co.uk format).
files = sorted(DATA_DIR.glob("*.csv"))
if not files:
    raise FileNotFoundError("No CSV files found. Upload your historical season CSVs into the data/ folder.")

frames = []
for path in files:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(frame.columns):
        print(f"Skipping {path.name}: missing one or more of {sorted(required)}")
        continue
    if "Date" not in frame.columns:
        raise ValueError(f"{path.name} has no Date column. Please use match data with dates.")
    if "FTR" not in frame.columns:
        frame["FTR"] = np.where(frame["FTHG"] > frame["FTAG"], "H",
                         np.where(frame["FTHG"] < frame["FTAG"], "A", "D"))
    frame = frame[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    frame["source_file"] = path.name
    frames.append(frame)

if not frames:
    raise ValueError("No usable CSV files found. Check that they are English Premier League match-result files.")

matches = pd.concat(frames, ignore_index=True)
# Football-data.co.uk files commonly use day/month/year. Parse robustly.
matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, errors="coerce")
matches["FTHG"] = pd.to_numeric(matches["FTHG"], errors="coerce")
matches["FTAG"] = pd.to_numeric(matches["FTAG"], errors="coerce")
matches["HomeTeam"] = matches["HomeTeam"].astype("string").str.strip()
matches["AwayTeam"] = matches["AwayTeam"].astype("string").str.strip()
matches["FTR"] = matches["FTR"].astype("string").str.strip().str.upper()
matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
matches = matches[matches["FTR"].isin(["H", "D", "A"])]
matches = matches[(matches["HomeTeam"] != "") & (matches["AwayTeam"] != "")]
matches = matches.sort_values(["Date", "source_file"]).reset_index(drop=True)
matches = matches.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
print(f"Loaded {len(matches):,} matches from {len(files)} CSV file(s).")
print("Date range:", matches["Date"].min().date(), "to", matches["Date"].max().date())

# 2) Make features strictly from matches dated BEFORE the current match date.
#    All fixtures on the same date are featurized before any results from that
#    date are added, avoiding same-day result leakage.
team_history = defaultdict(lambda: deque(maxlen=5))
team_totals = defaultdict(lambda: {"played": 0, "gf": 0, "ga": 0, "points": 0})
feature_rows = []

def team_snapshot(team):
    h = team_history[team]
    totals = team_totals[team]
    played = totals["played"]
    recent = list(h)
    # Most recent match is at the end of the deque.
    points5 = sum(x["points"] for x in recent)
    gf5 = sum(x["gf"] for x in recent)
    ga5 = sum(x["ga"] for x in recent)
    return {
        "played_before": played,
        "points_per_match": totals["points"] / played if played else 0.0,
        "goals_for_avg": totals["gf"] / played if played else 0.0,
        "goals_against_avg": totals["ga"] / played if played else 0.0,
        "points_last5": points5,
        "goals_for_last5": gf5,
        "goals_against_last5": ga5,
        "form_points_last5": points5 / max(len(recent), 1),
    }

for match_date, day_matches in matches.groupby("Date", sort=True):
    # Build every feature vector for this date before updating state.
    pending_updates = []
    for row in day_matches.itertuples(index=False):
        hs = team_snapshot(row.HomeTeam)
        aws = team_snapshot(row.AwayTeam)
        features = {
            "HomeTeam": row.HomeTeam,
            "AwayTeam": row.AwayTeam,
            **{f"home_{k}": v for k, v in hs.items()},
            **{f"away_{k}": v for k, v in aws.items()},
        }
        feature_rows.append({**features, "Date": match_date, "FTR": row.FTR})
        home_points = 3 if row.FTR == "H" else 1 if row.FTR == "D" else 0
        away_points = 3 if row.FTR == "A" else 1 if row.FTR == "D" else 0
        pending_updates.append((row.HomeTeam, row.AwayTeam, int(row.FTHG), int(row.FTAG),
                                home_points, away_points))
    # Only now update history with completed results.
    for home, away, hg, ag, hp, ap in pending_updates:
        team_history[home].append({"gf": hg, "ga": ag, "points": hp})
        team_history[away].append({"gf": ag, "ga": hg, "points": ap})
        team_totals[home]["played"] += 1
        team_totals[home]["gf"] += hg
        team_totals[home]["ga"] += ag
        team_totals[home]["points"] += hp
        team_totals[away]["played"] += 1
        team_totals[away]["gf"] += ag
        team_totals[away]["ga"] += hg
        team_totals[away]["points"] += ap

data = pd.DataFrame(feature_rows).sort_values("Date").reset_index(drop=True)
numeric_features = [c for c in data.columns if c.startswith(("home_", "away_"))]
categorical_features = ["HomeTeam", "AwayTeam"]
feature_cols = categorical_features + numeric_features

# Early matches have little historical form; keep them, with zeros meaning no
# prior match history. Split chronologically: last 20% is held out as test data.
split_at = int(len(data) * 0.80)
train_df = data.iloc[:split_at].copy()
test_df = data.iloc[split_at:].copy()
if train_df["FTR"].nunique() < 3:
    raise ValueError("Training period does not contain all three classes (H/D/A). Add more seasons.")

preprocessor = ColumnTransformer([
    ("teams", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ]), numeric_features)
])

candidates = {
    "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
}
results = []
best_name, best_pipeline, best_logloss = None, None, float("inf")

for name, estimator in candidates.items():
    pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipeline.fit(train_df[feature_cols], train_df["FTR"])
    pred = pipeline.predict(test_df[feature_cols])
    proba = pipeline.predict_proba(test_df[feature_cols])
    labels = list(pipeline.named_steps["model"].classes_)
    score = log_loss(test_df["FTR"], proba, labels=labels)
    acc = accuracy_score(test_df["FTR"], pred)
    results.append((name, acc, score))
    print(f"\n{name}: accuracy={acc:.4f}, log_loss={score:.4f}")
    print(classification_report(test_df["FTR"], pred, labels=["H", "D", "A"], zero_division=0))
    print("Confusion matrix (labels H, D, A):\n", confusion_matrix(test_df["FTR"], pred, labels=["H", "D", "A"]))
    if score < best_logloss:
        best_name, best_pipeline, best_logloss = name, pipeline, score

print("\nModel comparison (lower log loss is better for probabilities):")
print(pd.DataFrame(results, columns=["model", "accuracy", "log_loss"]).sort_values("log_loss"))

# 3) Save latest team snapshots for the prediction app.
# State was updated through all available historical results. This is a
# historical baseline only; it is not a live/current-season data feed.
latest_states = {}
for team in sorted(set(matches["HomeTeam"]).union(set(matches["AwayTeam"]))):
    snap = team_snapshot(team)
    latest_states[team] = snap

bundle = {
    "model": best_pipeline,
    "model_name": best_name,
    "feature_cols": feature_cols,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "team_states": latest_states,
    "teams": sorted(latest_states.keys()),
    "trained_through": str(matches["Date"].max().date()),
    "test_accuracy": float(dict((n, a) for n, a, _ in results)[best_name]),
    "test_log_loss": float(best_logloss),
}
joblib.dump(bundle, ARTIFACT_DIR / "football_model.joblib")
print(f"\nSaved {ARTIFACT_DIR / 'football_model.joblib'}")
print("Selected model:", best_name)
print("Historical data ends:", bundle["trained_through"])
print("Reminder: update the CSV dataset with recent seasons before using this for current fixtures.")
