import pandas as pd
from catboost import CatBoostRegressor

# from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

# from xgboost import XGBRegressor

from bike_sharing_model.config.core import RANDOM_STATE
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline
from bike_sharing_model.utils.helpers import create_train_test_df, evaluation_metrics


def model_accuracy(df: pd.DataFrame) -> dict[str, dict]:

    X_train, X_test, y_train, y_test = create_train_test_df(df)

    rush_transformer, preprocessor = create_preprocessing_pipeline()

    # models = {
    #     "XGBRegressor": XGBRegressor(
    #         random_state=RANDOM_STATE, n_estimators=300, learning_rate=0.05
    #     ),
    #     "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE),
    #     "LGBMRegressor": LGBMRegressor(random_state=RANDOM_STATE, n_estimators=300),
    # }

    models = {
        "CatBoostRegressor": CatBoostRegressor(verbose=0, random_state=RANDOM_STATE),
    }

    results: dict[str, dict] = {}

    for model_name, model in models.items():
        # Full model
        pipeline = Pipeline(
            [
                ("rush_hours", rush_transformer),
                ("preprocessing", preprocessor),
                ("model", model),
            ]
        )
        # Fitting it
        pipeline.fit(X_train, y_train)

        # Get predictions for train and test
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        # Dictionary of scores metrics
        results[model_name] = evaluation_metrics(
            y_test, y_pred_test, y_train, y_pred_train, end_point=True
        )

    return results
