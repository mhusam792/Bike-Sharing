import joblib
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from bike_sharing_model.config.core import (
    RANDOM_STATE,
    TESTING_DATA_FILE_PATH,
    TRAINED_MODEL_PATH,
    TRAINING_DATA_FILE_PATH,
)
from bike_sharing_model.data.loader import load_dataframe
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline
from bike_sharing_model.utils.helpers import create_train_test_df

from typing import Optional, Dict, Any


def create_best_model(
    df: pd.DataFrame,
    save_path=TRAINED_MODEL_PATH,
) -> dict:

    result = dict()

    X_train, X_test, y_train, y_test = create_train_test_df(df=df)

    test_df = X_test.copy()
    test_df["cnt"] = y_test.values
    test_df.to_csv(TESTING_DATA_FILE_PATH, index=False)

    result["test_csv_path"] = str(TESTING_DATA_FILE_PATH)

    rush_transformer, ct = create_preprocessing_pipeline()

    best_model_pipeline = Pipeline(
        [
            ("rush_hrs", rush_transformer),
            ("preprocessing", ct),
            ("model", CatBoostRegressor(verbose=0, random_state=RANDOM_STATE)),
        ]
    )

    best_model_pipeline.fit(X_train, y_train)

    joblib.dump(best_model_pipeline, f"{save_path}")
    result["saved_model_path"] = str(save_path)

    return result


def run_training(
    end_point: bool = False, com_models: bool = False
) -> Optional[Dict[str, Any]]:

    df = load_dataframe(path=TRAINING_DATA_FILE_PATH)

    result: Dict[str, Any] = {}

    # Always train best model
    best_model_info = create_best_model(df)
    result["best_model_info"] = best_model_info

    if not end_point:
        print(pd.DataFrame.from_dict(best_model_info, orient="index"))
        return None

    return result
