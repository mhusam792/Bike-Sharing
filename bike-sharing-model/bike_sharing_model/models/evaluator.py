import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from bike_sharing_model.utils.helpers import create_train_test_df
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline
from bike_sharing_model.config.settings import RANDOM_STATE

def evaluation_metrics(y_train, y_pred_train, y_test, y_pred_test) -> pd.DataFrame:
    return pd.DataFrame([{
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "rmse_train": root_mean_squared_error(y_train, y_pred_train),
        "rmse_test": root_mean_squared_error(y_test, y_pred_test),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
    }])


def compare_between_models(df: pd.DataFrame) -> pd.DataFrame:

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

    return (
        pd.concat(results)
        .set_index("model")
        .sort_values("rmse_test")
    )
