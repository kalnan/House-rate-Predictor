"""
app.py
------
Streamlit app for predicting house/flat prices in Kolkata, India.
Adapted from house-price-prediction.ipynb, but re-scoped end to end for
the Kolkata market:

  - Locality dropdown uses real Kolkata areas (Ballygunge, Salt Lake,
    New Town, Behala, etc.) instead of the original Ames, Iowa
    neighborhoods.
  - Inputs use fields that actually drive Kolkata listings: BHK
    (Bedroom-Hall-Kitchen), floor number / total floors, lift
    availability, car parking, balconies -- instead of US-specific
    fields like basement sq ft or fireplaces.
  - Output is in Indian Rupees, shown with Indian digit grouping and a
    Lakh/Crore label (e.g. "Rs 1,25,00,000 (~Rs 1.25 Crore)").

No real Kolkata listings dataset was supplied, so model.pkl is trained on
a synthetic dataset (see train_model.py) with per-locality price-per-sqft
rates calibrated to be roughly realistic. Swap in a real dataset there
when you have one -- app.py itself needs no changes.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import train_model

MODEL_PATH = "model.pkl"

st.set_page_config(page_title="Kolkata House Price Predictor", page_icon="🏠", layout="centered")


def format_inr(amount: float) -> str:
    """Format a number with Indian digit grouping, e.g. 12500000 -> '1,25,00,000'."""
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    s = str(amount)
    if len(s) <= 3:
        return sign + s
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return sign + ",".join(groups) + "," + last3


def lakh_crore_label(amount: float) -> str:
    if amount >= 1_00_00_000:
        return f"~₹{amount / 1_00_00_000:,.2f} Crore"
    return f"~₹{amount / 1_00_000:,.2f} Lakh"


REQUIRED_KEYS = {"pipeline", "numeric_features", "categorical_features", "localities"}


@st.cache_resource
def load_model():
    def _train():
        with st.spinner("Training model (first run, or outdated model.pkl found)..."):
            train_model.main()

    if not os.path.exists(MODEL_PATH):
        _train()

    def _load_valid():
        # Raises on a corrupt/version-mismatched pickle, and also on a
        # *stale* model.pkl left over from an earlier, differently-shaped
        # version of this app (e.g. the old US-features build, which has
        # no "localities" key) -- both cases should trigger a retrain
        # rather than crashing later on a KeyError.
        loaded = joblib.load(MODEL_PATH)
        if not REQUIRED_KEYS.issubset(loaded.keys()):
            raise ValueError("model.pkl is from an older/incompatible app version")
        return loaded

    try:
        return _load_valid()
    except Exception:
        _train()
        try:
            return _load_valid()
        except Exception:
            st.error(
                "Couldn't load a valid model even after retraining.\n\n"
                "This almost always means **train_model.py** in this folder "
                "is an older version that doesn't match app.py (so "
                "retraining just regenerates the same incompatible "
                "model.pkl).\n\n"
                "Fix: make sure both **app.py** and **train_model.py** are "
                "the latest versions saved in the same folder, delete "
                "**model.pkl**, and rerun `streamlit run app.py`."
            )
            st.stop()


bundle = load_model()
pipeline = bundle["pipeline"]
localities = bundle["localities"]

st.title("🏠 Kolkata House Price Predictor")
st.caption(
    "Estimate a property's sale price in Kolkata from its key characteristics "
    "(adapted from the House Price EDA & Prediction notebook)."
)

st.warning(
    "⚠️ This app's model is trained on **synthetic** data (no real Kolkata "
    "listings dataset was supplied), so predictions illustrate the app's "
    "mechanics rather than actual market prices. Swap in a real Kolkata "
    "dataset in train_model.py for genuine predictions.",
    icon="⚠️",
)

with st.form("prediction_form"):
    st.subheader("Location")
    locality = st.selectbox("Locality", sorted(localities))

    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    with col1:
        bhk = st.selectbox("BHK (Bedroom-Hall-Kitchen)", [1, 2, 3, 4, 5], index=2)
        builtup_area = st.number_input("Built-up area (sq ft)", 250, 6000, 1200, step=50)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
    with col2:
        balconies = st.selectbox("Balconies", [0, 1, 2, 3, 4], index=1)
        car_parking = st.selectbox("Car parking spaces", [0, 1, 2, 3], index=1)
        plot_area = st.number_input(
            "Plot area (sq ft) - independent house only, 0 for flat/apartment",
            0, 8000, 0, step=100,
        )

    st.subheader("Building Details")
    col1, col2 = st.columns(2)
    with col1:
        total_floors = st.number_input("Total floors in building", 1, 40, 6)
        floor_number = st.number_input("Floor number (0 = ground floor)", 0, 39, 2)
    with col2:
        has_lift = st.radio("Lift available?", ["Yes", "No"], horizontal=True) == "Yes"
        year_built = st.number_input("Year built", 1950, 2026, 2015, step=1)

    st.subheader("Quality & Condition")
    col1, col2 = st.columns(2)
    with col1:
        overall_qual = st.slider("Construction quality (1=Poor, 10=Excellent)", 1, 10, 6)
    with col2:
        overall_cond = st.slider("Property condition (1=Poor, 10=Excellent)", 1, 10, 6)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    if floor_number > total_floors - 1:
        st.error("Floor number can't exceed the total floors in the building.")
    else:
        input_df = pd.DataFrame([{
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "BuiltupArea": builtup_area,
            "BHK": bhk,
            "Bathrooms": bathrooms,
            "Balconies": balconies,
            "FloorNumber": floor_number,
            "TotalFloors": total_floors,
            "HasLift": int(has_lift),
            "CarParking": car_parking,
            "PlotArea": plot_area,
            "YearBuilt": year_built,
            "Locality": locality,
        }])

        log_pred = pipeline.predict(input_df)[0]
        price = float(np.expm1(log_pred))

        st.success(f"### Estimated Price: ₹{format_inr(price)} ({lakh_crore_label(price)})")

        margin = price * 0.10
        st.caption(
            f"Typical range: ₹{format_inr(price - margin)} – ₹{format_inr(price + margin)}"
        )

st.divider()
with st.expander("About this app"):
    st.markdown(
        "- Model: Gradient Boosting Regressor trained on a log-transformed "
        "target (property price is right-skewed, same rationale as the "
        "notebook's `np.log1p(SalePrice)` step).\n"
        "- Localities and per-sq-ft rates are illustrative approximations "
        "of the Kolkata market (Alipore/Ballygunge premium, New Town/Salt "
        "Lake mid-range, outer suburbs like Barasat/Kestopur more "
        "affordable) -- not an official valuation source.\n"
        "- To use real data: replace `generate_synthetic_data()` in "
        "`train_model.py` with an actual Kolkata listings dataset (same "
        "column names), delete `model.pkl`, and rerun."
    )
