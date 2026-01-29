import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from bike_sharing_model.config.core import (
    RANDOM_STATE,
    TESTING_DATA_FILE_PATH,
    TRAINING_DATA_FILE_PATH,
)
from bike_sharing_model.data.loader import load_dataframe
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline
from bike_sharing_model.utils.helpers import (
    create_train_test_df,
    evaluation_metrics,
    reshape_comparing_df,
)
from bike_sharing_model.models.evaluator import model_accuracy
from bike_sharing_model.features.feature_engineering import RushHourTransformer

from typing import Optional, Dict, Any
import dagshub
import mlflow


def create_best_model(
    df: pd.DataFrame,
) -> dict:

    result = {}

    # dagshub.init(repo_owner="Mohamed_Hussam", repo_name="Bike-Sharing", mlflow=True)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("bike-sharing-training")
    with mlflow.start_run(run_name="best-model-training"):

        X_train, X_test, y_train, y_test = create_train_test_df(df=df)

        test_df = X_test.copy()
        test_df["cnt"] = y_test.values
        test_df.to_csv(TESTING_DATA_FILE_PATH, index=False)

        result["test_csv_path"] = str(TESTING_DATA_FILE_PATH)
        mlflow.log_artifact(result["test_csv_path"])

        ct = create_preprocessing_pipeline()

        best_model_pipeline = Pipeline(
            [
                (
                    "rush_hours",
                    RushHourTransformer(variables=["hr"], target="cnt", top_n=5),
                ),
                ("preprocessing", ct),
                (
                    "model",
                    CatBoostRegressor(
                        verbose=0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        best_model_pipeline.fit(X_train, y_train)

        y_pred_train = best_model_pipeline.predict(X_train)
        y_pred_test = best_model_pipeline.predict(X_test)

        metrics = evaluation_metrics(
            y_test, y_pred_test, y_train, y_pred_train, end_point=True
        )

        mlflow.log_metrics(
            {
                "r2_test": metrics["test_score"]["r2"],
                "rmse_test": metrics["test_score"]["rmse"],
                "mae_test": metrics["test_score"]["mae"],
                "r2_train": metrics["train_score"]["r2"],
                "rmse_train": metrics["train_score"]["rmse"],
                "mae_train": metrics["train_score"]["mae"],
            }
        )

        mlflow.log_params(
            {
                "model": "CatBoostRegressor",
                "random_state": RANDOM_STATE,
                "rush_hour_top_n": 5,
            }
        )

        mlflow.sklearn.log_model(
            sk_model=best_model_pipeline,
            name="best_model",
            registered_model_name="bike_sharing_demand_model",
        )

    return result


def run_training(
    end_point: bool = False, show_accuracy: bool = False
) -> Optional[Dict[str, Any]]:

    df = load_dataframe(path=TRAINING_DATA_FILE_PATH)

    result: Dict[str, Any] = {}

    # Compare models if requested
    if show_accuracy:
        model_acc = model_accuracy(df)
        reshape_comp_df = reshape_comparing_df(model_acc)

        result["model_accuracy"] = model_acc

        if not end_point:
            print(pd.DataFrame(reshape_comp_df).set_index(["model", "split"]))

    # Always train best model
    best_model_info = create_best_model(df)
    result["best_model_info"] = best_model_info

    if not end_point:
        print(pd.DataFrame.from_dict(best_model_info, orient="index"))
        return None

    return result
