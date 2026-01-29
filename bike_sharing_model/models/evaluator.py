from lightgbm import LGBMRegressor
import pandas as pd
from catboost import CatBoostRegressor

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from bike_sharing_model.config.core import RANDOM_STATE
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline
from bike_sharing_model.features.feature_engineering import RushHourTransformer
from bike_sharing_model.utils.helpers import create_train_test_df, evaluation_metrics

import mlflow
import dagshub


def model_accuracy(df: pd.DataFrame) -> dict[str, dict]:

    X_train, X_test, y_train, y_test = create_train_test_df(df)

    preprocessor = create_preprocessing_pipeline()

    models = {
        "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE),
        "XGBRegressor": XGBRegressor(
            random_state=RANDOM_STATE, n_estimators=300, learning_rate=0.05
        ),
        "LGBMRegressor": LGBMRegressor(random_state=RANDOM_STATE, n_estimators=300),
    }

    # models = {
    #     "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE),
    # }

    results: dict[str, dict] = {}

    # mlflow.set_tracking_uri("http://localhost:5000")
    dagshub.init(repo_owner="Mohamed_Hussam", repo_name="Bike-Sharing", mlflow=True)
    mlflow.set_experiment("bike-sharing-training")
    with mlflow.start_run(run_name="Compare models"):

        for model_name, model in models.items():
            # Full model
            pipeline = Pipeline(
                [
                    (
                        "rush_hours",
                        RushHourTransformer(variables=["hr"], target="cnt", top_n=5),
                    ),
                    ("preprocessing", preprocessor),
                    ("model", model),
                ]
            )

            # Fitting it
            pipeline.fit(X_train, y_train)
            model_info = mlflow.sklearn.log_model(sk_model=pipeline, name=model_name)

            # Get predictions for train and test
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)

            # Dictionary of scores metrics
            results[model_name] = evaluation_metrics(
                y_test, y_pred_test, y_train, y_pred_train, end_point=True
            )

            loged_metrices = {
                "r2_test": results[model_name]["test_score"]["r2"],
                "rmse_test": results[model_name]["test_score"]["rmse"],
                "mae_test": results[model_name]["test_score"]["mae"],
                "r2_train": results[model_name]["train_score"]["r2"],
                "rmse_train": results[model_name]["train_score"]["rmse"],
                "mae_train": results[model_name]["train_score"]["mae"],
            }

            mlflow.log_metrics(metrics=loged_metrices)

        return results
