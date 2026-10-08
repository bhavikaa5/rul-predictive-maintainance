"""Streamlit demo: streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import CRITICAL_BELOW, MODELS_DIR, WARNING_BELOW
from src.data import load_raw
from src.features import apply_scaler, trajectory_windows

st.set_page_config(page_title="Engine RUL", layout="wide")


@st.cache_resource
def load_artifacts():
    from tensorflow import keras
    meta = json.load(open(MODELS_DIR / "meta.json"))
    return keras.models.load_model(MODELS_DIR / "best_model.keras"), joblib.load(MODELS_DIR / "scaler.pkl"), meta


@st.cache_data
def load_test():
    _, test, rul = load_raw()
    return test, rul


try:
    model, scaler, meta = load_artifacts()
    test, rul_test = load_test()
except Exception as e:  # missing models / data
    st.error(f"Could not load model or data: {e}\nRun `python -m src.run_all` (or notebook 02) first.")
    st.stop()

feats, window = meta["features"], meta["window"]
st.title("Jet engine Remaining Useful Life")
st.caption(f"Model: {meta['best_model']} (window {window}) trained on NASA C-MAPSS FD001")

engine = st.sidebar.selectbox("Test engine", sorted(test.engine.unique()))
crit = st.sidebar.number_input("Critical below (cycles)", value=CRITICAL_BELOW)
warn = st.sidebar.number_input("Warning below (cycles)", value=WARNING_BELOW)
sensor = st.sidebar.multiselect("Sensors to plot", feats, default=feats[:4])

g = test[test.engine == engine]
scaled = apply_scaler(g, scaler, feats)
pred = np.maximum(model.predict(trajectory_windows(scaled, window, feats), verbose=0).ravel(), 0)
now = float(pred[-1])
status = "Critical" if now < crit else "Warning" if now < warn else "Healthy"
color = {"Critical": "red", "Warning": "orange", "Healthy": "green"}[status]

c1, c2, c3 = st.columns(3)
c1.metric("Predicted RUL (cycles)", f"{now:.0f}")
c2.metric("Cycles observed", len(g))
c3.markdown(f"### Status: :{color}[{status}]")
st.caption(f"True RUL at last cycle (from RUL_FD001.txt): {rul_test[engine]}")

left, right = st.columns(2)
with left:
    st.subheader("Sensor history")
    fig, ax = plt.subplots()
    for s in sensor:
        ax.plot(g.cycle, g[s], label=s)
    ax.set(xlabel="cycle"); ax.legend()
    st.pyplot(fig)
with right:
    st.subheader("Predicted RUL over time")
    fig, ax = plt.subplots()
    ax.plot(g.cycle, pred)
    ax.axhline(crit, color="red", ls="--", lw=1); ax.axhline(warn, color="orange", ls="--", lw=1)
    ax.set(xlabel="cycle", ylabel="predicted RUL")
    st.pyplot(fig)
