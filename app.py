from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "ai4i2020.csv"
DB_PATH = BASE_DIR / "maintenance.db"
MODEL_PATH = BASE_DIR / "model.joblib"
CSV_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
NUMERIC_FEATURES = FEATURES[1:]


def load_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        try:
            response = requests.get(CSV_URL, timeout=20)
            response.raise_for_status()
            DATA_PATH.write_bytes(response.content)
        except Exception as exc:
            raise RuntimeError(
                "Dataset not found and automatic download failed. Download ai4i2020.csv "
                "from the UCI AI4I 2020 dataset page and save it in the project's data folder. "
                f"Details: {exc}"
            )
    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def build_model(df):
    # Exclude UID, Product ID, and TWF/HDF/PWF/OSF/RNF: identifiers are not useful
    # here and failure-mode flags reveal the target (target leakage).
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=180,
                class_weight="balanced",
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            )),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1 score": f1_score(y_test, predicted, zero_division=0),
    }
    cm = confusion_matrix(y_test, predicted, labels=[0, 1])
    return model, X_test.reset_index(drop=True), y_test.reset_index(drop=True), metrics, cm


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                machine_type TEXT NOT NULL,
                air_temperature_k REAL NOT NULL,
                process_temperature_k REAL NOT NULL,
                rotational_speed_rpm REAL NOT NULL,
                torque_nm REAL NOT NULL,
                tool_wear_min REAL NOT NULL,
                failure_probability REAL NOT NULL,
                predicted_failure INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                actual_failure INTEGER
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reading_id INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(reading_id) REFERENCES readings(id)
            )
        """)


def risk_label(probability):
    # Demonstration thresholds; tune using validation data and maintenance costs.
    if probability >= 0.60:
        return "CRITICAL"
    if probability >= 0.30:
        return "WARNING"
    return "NORMAL"


def record_reading(row, probability, actual_failure=None):
    level = risk_label(probability)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("""
            INSERT INTO readings (
                recorded_at, machine_type, air_temperature_k, process_temperature_k,
                rotational_speed_rpm, torque_nm, tool_wear_min,
                failure_probability, predicted_failure, risk_level, actual_failure
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, str(row["Type"]), float(row["Air temperature [K]"]),
            float(row["Process temperature [K]"]), float(row["Rotational speed [rpm]"]),
            float(row["Torque [Nm]"]), float(row["Tool wear [min]"]),
            float(probability), int(probability >= 0.50), level,
            None if actual_failure is None else int(actual_failure),
        ))
        reading_id = cur.lastrowid
        if level in ("WARNING", "CRITICAL"):
            message = (
                "Elevated model-estimated failure risk. Inspect the machine and verify "
                "sensor readings before taking maintenance action."
            )
            conn.execute("""
                INSERT INTO maintenance_events (reading_id, recorded_at, risk_level, message)
                VALUES (?, ?, ?, ?)
            """, (reading_id, now, level, message))
    return level


def read_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM readings ORDER BY id DESC", conn)


st.set_page_config(page_title="Industrial Predictive Maintenance", page_icon="🏭", layout="wide")
st.title("🏭 Real-Time Industrial Predictive Maintenance")
st.caption("Educational demonstration using the AI4I 2020 dataset, a Random Forest model, and SQLite.")

try:
    with st.spinner("Loading dataset and training the model..."):
        data = load_dataset()
        model, X_test, y_test, metrics, cm = build_model(data)
        init_db()
except Exception as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("Simulation controls")
    st.write(f"Dataset rows: **{len(data):,}**")
    st.write(f"Recorded readings: **{len(read_history()):,}**")
    st.info("Use the buttons below to simulate new sensor readings from held-out dataset rows.")
    if "stream_index" not in st.session_state:
        st.session_state.stream_index = 0
    if st.button("Next reading", use_container_width=True):
        idx = st.session_state.stream_index % len(X_test)
        row = X_test.iloc[idx]
        prob = float(model.predict_proba(pd.DataFrame([row]))[0][1])
        record_reading(row, prob, int(y_test.iloc[idx]))
        st.session_state.stream_index += 1
        st.rerun()
    if st.button("Run 10 readings", use_container_width=True):
        for _ in range(10):
            idx = st.session_state.stream_index % len(X_test)
            row = X_test.iloc[idx]
            prob = float(model.predict_proba(pd.DataFrame([row]))[0][1])
            record_reading(row, prob, int(y_test.iloc[idx]))
            st.session_state.stream_index += 1
        st.rerun()
    if st.button("Clear monitoring history", use_container_width=True):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM maintenance_events")
            conn.execute("DELETE FROM readings")
        st.rerun()

st.subheader("Model evaluation on held-out test data")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
m2.metric("Precision", f"{metrics['Precision']:.1%}")
m3.metric("Recall", f"{metrics['Recall']:.1%}")
m4.metric("F1 score", f"{metrics['F1 score']:.3f}")
st.caption("These metrics are computed from a held-out test split. Rare failures can make accuracy misleading; review precision, recall, and F1 together.")

cm_df = pd.DataFrame(cm, index=["Actual normal", "Actual failure"], columns=["Predicted normal", "Predicted failure"])
left, right = st.columns([1, 2])
with left:
    st.markdown("**Confusion matrix**")
    st.dataframe(cm_df, use_container_width=True)
with right:
    st.markdown("**Latest monitoring records**")
    history = read_history()
    if history.empty:
        st.info("No readings recorded yet. Click **Next reading** or **Run 10 readings** in the sidebar.")
    else:
        show_cols = [
            "recorded_at", "machine_type", "air_temperature_k", "process_temperature_k",
            "rotational_speed_rpm", "torque_nm", "tool_wear_min",
            "failure_probability", "risk_level"
        ]
        display = history[show_cols].head(15).copy()
        display["failure_probability"] = (display["failure_probability"] * 100).round(1).astype(str) + "%"
        st.dataframe(display, use_container_width=True, hide_index=True)

history = read_history()
if not history.empty:
    st.subheader("Monitoring trends")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            history.sort_values("id"), x="id", y="failure_probability",
            title="Model-estimated failure risk by reading",
            labels={"id": "Reading number", "failure_probability": "Estimated probability"},
        )
        fig.add_hline(y=0.30, line_dash="dash", annotation_text="Warning threshold")
        fig.add_hline(y=0.60, line_dash="dash", annotation_text="Critical threshold")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        counts = history["risk_level"].value_counts().rename_axis("Risk level").reset_index(name="Readings")
        fig2 = px.bar(counts, x="Risk level", y="Readings", title="Readings by risk category")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Maintenance alerts")
    alerts = history[history["risk_level"].isin(["WARNING", "CRITICAL"])].head(20)
    if alerts.empty:
        st.success("No warning or critical predictions in the recorded sample.")
    else:
        st.dataframe(
            alerts[["recorded_at", "risk_level", "failure_probability", "machine_type",
                    "rotational_speed_rpm", "torque_nm", "tool_wear_min"]],
            use_container_width=True, hide_index=True
        )

with st.expander("About the dataset and responsible use"):
    st.write(
        "The AI4I 2020 dataset is synthetic and contains operating features plus a machine-failure label. "
        "This app excludes the failure-mode columns (TWF, HDF, PWF, OSF, RNF) from model inputs to avoid target leakage. "
        "The app simulates real-time arrival by recording one held-out row per button click. It does not receive live sensor data. "
        "The score is a model estimate, not a guaranteed prediction of a future failure. Validate with real equipment data before operational use."
    )
    st.markdown("Dataset: [UCI Machine Learning Repository – AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)")
