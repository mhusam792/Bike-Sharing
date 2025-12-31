from typing import List, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from bike_sharing_model.config.core import (
    FEATURES_LIST,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def create_train_test_df(
    df: pd.DataFrame, features: List[str] | None = None, target: str | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    if features is None or target is None:
        features = FEATURES_LIST
        target = TARGET

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def evaluation_metrics(
    y_test, y_pred_test, y_train=None, y_pred_train=None, end_point: bool = False
) -> pd.DataFrame | dict:
    test_score = {
        "r2": r2_score(y_test, y_pred_test),
        "rmse": root_mean_squared_error(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
    }

    result = {"test_score": test_score}

    if (y_train is not None) and (y_pred_train is not None):
        train_score = {
            "r2": r2_score(y_train, y_pred_train),
            "rmse": root_mean_squared_error(y_train, y_pred_train),
            "mae": mean_absolute_error(y_train, y_pred_train),
        }
        result["train_score"] = train_score

    if end_point:
        return result
    return pd.DataFrame(result)


def reshape_comparing_df(comparing_dict: dict) -> pd.DataFrame:
    rows = []

    for model_name, scores in comparing_dict.items():
        for split in ["train", "test"]:
            score_key = f"{split}_score"
            metrics = scores[score_key]

            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                }
            )

    df_results = pd.DataFrame(rows)
    return df_results
