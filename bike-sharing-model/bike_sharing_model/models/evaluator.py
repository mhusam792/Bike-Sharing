import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from bike_sharing_model.utils.helpers import create_train_test_df, evaluation_metrics
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline
from bike_sharing_model.config.core import RANDOM_STATE


def compare_between_models(df: pd.DataFrame) -> list[dict]:

    X_train, X_test, y_train, y_test = create_train_test_df(df)

    rush_transformer, preprocessor = create_preprocessing_pipeline()

    models = {
        "XGBRegressor": XGBRegressor(
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05
        ),
        "CatBoostRegressor": CatBoostRegressor(
            verbose=0,
            random_state=RANDOM_STATE
        ),
        "LGBMRegressor": LGBMRegressor(
            random_state=RANDOM_STATE,
            n_estimators=300
        ),
    }

    results = []

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("rush_hours", rush_transformer),
            ("preprocessing", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        metrics_df = evaluation_metrics(
            y_train, y_pred_train, y_test, y_pred_test
        )
        metrics_df["model"] = model_name
        results.append(metrics_df)

    df = (
        pd.concat(results)
        .set_index("model")
        .sort_values("rmse_test")
    )

    return df.reset_index().to_dict(orient="records")
