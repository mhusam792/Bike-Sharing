
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error
)

from typing import List, Tuple

from bike_sharing_model.config.core import (
    FEATURES_LIST,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE
)


def create_train_test_df(
        df: pd.DataFrame, 
        features: List[str] | None=None, 
        target: str | None =None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    if features is None or target is None:
        features = FEATURES_LIST
        target = TARGET

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=TEST_SIZE, 
        shuffle=False, 
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test


def evaluation_metrics(y_train, y_pred_train, y_test, y_pred_test) -> pd.DataFrame:
    return pd.DataFrame([{
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_train": root_mean_squared_error(y_train, y_pred_train),
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
    }])