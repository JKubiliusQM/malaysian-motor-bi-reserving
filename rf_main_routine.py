""" Reproducible Random Forest future-development routine.

The routine shows the feature definition, target construction, temporal split,
preprocessing, model fitting, prediction, non-negative reconstruction and
model export used in the dissertation. File-discovery and extended audit code
are omitted from the printed appendix.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CATEGORICAL_FEATURES = [
    "CLASS",
    "cover",
    "NATLOSS",
    "ACC_QTR",
]

NUMERIC_FEATURES = [
    "ACC_YEAR",
    "PAIDLS",
    "BALOS",
    "TAG_INC_LARGE",
    "TAG_PAIDLS_LARGE",
    "CLAIMS_CNT",
    "SETTLED_CNT",
    "CUM_PAIDLS",
    "CUM_INC_AMT",
    "CUM_INC_NON_LARGE",
    "CUM_INC_LARGE",
    "CLAIMS_CNT_LARGE",
    "SETTLED_CNT_LARGE",
    "CUM_PAIDLS_NON_LARGE",
    "CUM_PAIDLS_LARGE",
    "is_bi_excess_at_snapshot",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_LATEST = "target_latest_CUM_INC_LARGE"
TARGET_FUTURE = "future_excess_incurred_development_clean"
SNAPSHOT_EXCESS = "snapshot_CUM_INC_LARGE"

TRAIN_LABEL = "train_up_to_2019_Q4"
TEST_LABEL = "test_2020_Q1_to_2022_Q4"

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 50,
    "min_samples_split": 2,
    "max_features": 1.0,
    "bootstrap": True,
    "criterion": "squared_error",
    "random_state": 42,
    "n_jobs": -1,
}


def make_one_hot_encoder() -> OneHotEncoder:
    """Create an encoder compatible with old and new scikit-learn releases."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def build_rf_pipeline() -> Pipeline:
    """Create the fitted preprocessing and Random Forest pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(**RF_PARAMS),
            ),
        ]
    )


def fit_snapshot_model(
    snapshot_df: pd.DataFrame,
    snapshot_name: str,
    output_folder: Path,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit and score one maturity-specific future-development model."""
    df = snapshot_df.copy()

    # A null excess value means that no BI Excess amount was recognised.
    df["CUM_INC_LARGE"] = pd.to_numeric(
        df["CUM_INC_LARGE"],
        errors="coerce",
    ).fillna(0.0)

    df[TARGET_LATEST] = pd.to_numeric(
        df[TARGET_LATEST],
        errors="coerce",
    ).fillna(0.0)

    # Construct the known snapshot position and future-development target.
    df[SNAPSHOT_EXCESS] = df["CUM_INC_LARGE"].astype(float)
    df[TARGET_FUTURE] = (
        df[TARGET_LATEST] - df[SNAPSHOT_EXCESS]
    )
    df["is_bi_excess_at_snapshot"] = (
        df[SNAPSHOT_EXCESS] > 0
    ).astype("int8")

    train_mask = df["valuation_split"].eq(TRAIN_LABEL)
    test_mask = df["valuation_split"].eq(TEST_LABEL)

    X_train = df.loc[train_mask, FEATURE_COLUMNS]
    X_test = df.loc[test_mask, FEATURE_COLUMNS]

    y_train = df.loc[train_mask, TARGET_FUTURE].astype(float)
    y_test = df.loc[test_mask, TARGET_FUTURE].astype(float)

    snapshot_test = df.loc[
        test_mask,
        SNAPSHOT_EXCESS,
    ].astype(float).to_numpy()

    model = build_rf_pipeline()
    model.fit(X_train, y_train)

    # Raw future development can be positive or negative.
    predicted_future_raw = model.predict(X_test)

    # The deployed later BI Excess amount cannot be negative.
    predicted_later_raw = (
        snapshot_test + predicted_future_raw
    )
    predicted_later_deployed = np.maximum(
        predicted_later_raw,
        0.0,
    )

    predictions = df.loc[
        test_mask,
        ["SOURCE_FILE", "CLAIMS_KEY", "ACC_YEAR", "ACC_QTR"],
    ].copy()

    predictions["snapshot"] = snapshot_name
    predictions["actual_future_development"] = y_test.to_numpy()
    predictions["predicted_future_development_raw"] = (
        predicted_future_raw
    )
    predictions["snapshot_bixs"] = snapshot_test
    predictions["predicted_later_bixs"] = (
        predicted_later_deployed
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        output_folder
        / f"rf_future_development_{snapshot_name}.joblib",
    )

    predictions.to_parquet(
        output_folder
        / f"rf_test_predictions_{snapshot_name}.parquet",
        index=False,
    )

    return model, predictions


# The same RF specification is fitted separately to each maturity dataset:
#
# for snapshot_name in ["DEV_QTR_4", "DEV_QTR_8", "DEV_QTR_12"]:
#     snapshot_df = pd.read_parquet(f"{snapshot_name}_model_ready.parquet")
#     fit_snapshot_model(snapshot_df, snapshot_name, OUTPUT_FOLDER)
