"""
train_model.py
---------------
Builds the model used by app.py -- customized for the Kolkata, India
housing market.

The original notebook (house-price-prediction.ipynb) trains on the Kaggle
"House Prices - Advanced Regression Techniques" dataset (Ames, Iowa, USD,
~80 raw features). That dataset/pricing structure doesn't map onto how
property is actually described or priced in Kolkata, so this version:

  - Replaces the US "Neighborhood" categories with real Kolkata localities,
    each carrying a realistic per-sq-ft rate (INR).
  - Replaces US-style fields (basement sq ft, fireplaces, garage sq ft)
    with fields that actually drive Kolkata property listings: BHK
    (Bedroom-Hall-Kitchen configuration), floor number / total floors,
    lift availability, car parking, balconies, plot area for independent
    houses.
  - Generates a synthetic dataset (no real Kolkata dataset was supplied)
    whose price relationships are calibrated in INR (roughly 15 lakh to
    a few crore, varying by locality) instead of USD.
  - Still trains in log1p space (matches the notebook's approach, since
    property prices are right-skewed) and uses the same
    ColumnTransformer + GradientBoosting pipeline structure.

--> Swap generate_synthetic_data() for a real Kolkata listings CSV the
    moment you have one; app.py needs no changes since it consumes
    model.pkl's feature list.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

RANDOM_STATE = 42
N_SAMPLES = 3000
CURRENT_YEAR = 2026

# Kolkata localities with an approximate market rate (INR per sq ft of
# built-up area). These are illustrative, roughly-ordered figures (posh
# South/Central Kolkata > developing New Town/Rajarhat > outer suburbs),
# not an official valuation source.
LOCALITIES = {
    "Alipore": 15500,
    "Ballygunge": 15000,
    "Park Street Area": 13500,
    "Southern Avenue": 11500,
    "Bhowanipore": 11000,
    "Gariahat": 10500,
    "Salt Lake (Bidhannagar)": 9500,
    "New Town": 8500,
    "Tollygunge": 7500,
    "Jadavpur": 7200,
    "Kasba": 7200,
    "Chetla": 8000,
    "Behala": 5800,
    "Garia": 5800,
    "Rajarhat": 6800,
    "Lake Town": 6500,
    "Dum Dum": 5500,
    "Baguiati": 5200,
    "Kestopur": 4800,
    "Howrah": 5000,
    "Barasat": 4000,
}
LOCALITY_NAMES = list(LOCALITIES.keys())

NUMERIC_FEATURES = [
    "OverallQual", "OverallCond", "BuiltupArea", "BHK", "Bathrooms",
    "Balconies", "FloorNumber", "TotalFloors", "HasLift", "CarParking",
    "PlotArea", "YearBuilt",
]
CATEGORICAL_FEATURES = ["Locality"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def generate_synthetic_data(n=N_SAMPLES, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    overall_qual = rng.integers(2, 11, n)          # construction quality
    overall_cond = rng.integers(2, 10, n)           # upkeep/condition
    bhk = rng.integers(1, 6, n)                      # 1BHK - 5BHK
    builtup_area = np.clip(
        bhk * rng.normal(420, 90, n) + rng.normal(150, 60, n), 300, 4500
    )
    bathrooms = np.clip(bhk - rng.integers(0, 2, n), 1, 6)
    balconies = np.clip(bhk - rng.integers(0, 2, n), 0, 5)
    total_floors = rng.integers(2, 20, n)
    floor_number = np.array([rng.integers(0, tf) for tf in total_floors])
    has_lift = (total_floors >= 4).astype(int)
    has_lift = np.where(rng.random(n) < 0.08, 1 - has_lift, has_lift)  # some noise
    car_parking = rng.integers(0, 3, n)
    # Only some properties are independent houses with extra land;
    # most are flats with PlotArea = 0.
    is_independent_house = rng.random(n) < 0.25
    plot_area = np.where(
        is_independent_house, np.clip(rng.normal(1200, 500, n), 400, 6000), 0
    )
    year_built = rng.integers(1975, CURRENT_YEAR, n)
    locality = rng.choice(LOCALITY_NAMES, n)
    rate_per_sqft = np.array([LOCALITIES[loc] for loc in locality])

    df = pd.DataFrame({
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "BuiltupArea": builtup_area.round(0),
        "BHK": bhk,
        "Bathrooms": bathrooms,
        "Balconies": balconies,
        "FloorNumber": floor_number,
        "TotalFloors": total_floors,
        "HasLift": has_lift,
        "CarParking": car_parking,
        "PlotArea": plot_area.round(0),
        "YearBuilt": year_built,
        "Locality": locality,
    })

    quality_mult = 0.70 + 0.06 * overall_qual          # 0.82 - 1.30
    condition_mult = 0.85 + 0.03 * overall_cond         # 0.91 - 1.15
    age = CURRENT_YEAR - year_built
    age_mult = np.clip(1.05 - 0.004 * age, 0.65, 1.05)  # newer = pricier

    # floor premium: very high or ground floor is slightly less desirable
    # unless there's a lift; mid floors carry a small premium.
    floor_mult = 1.0 + np.where(
        (floor_number == 0) & (has_lift == 0), -0.03,
        np.where(floor_number >= 8, 0.02, 0.0)
    )

    base_price = (
        rate_per_sqft * df["BuiltupArea"].values
        * quality_mult * condition_mult * age_mult * floor_mult
    )

    addons = (
        car_parking * 200_000
        + bathrooms * 80_000
        + balconies * 40_000
        + has_lift * 150_000
        + bhk * 50_000
        + plot_area * 3_000
    )

    noise = rng.normal(0, 0.08, n)
    price = (base_price + addons) * np.exp(noise)
    price = np.clip(price, 1_500_000, 80_000_000)  # sanity bounds: 15L - 8Cr

    df["SalePrice"] = price.round(0)

    return df


def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main():
    df = generate_synthetic_data()

    X = df[ALL_FEATURES]
    y_log = np.log1p(df["SalePrice"])  # price is right-skewed, same as notebook

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    pred_log = pipeline.predict(X_test)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    print(f"Validation MAE: Rs {mae:,.0f}")
    print(f"Validation R^2: {r2:.3f}")

    joblib.dump(
        {
            "pipeline": pipeline,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "localities": LOCALITY_NAMES,
        },
        "model.pkl",
    )
    print("Saved model.pkl")


if __name__ == "__main__":
    main()
