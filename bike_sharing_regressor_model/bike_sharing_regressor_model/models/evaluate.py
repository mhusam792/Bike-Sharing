import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from bike_sharing_regressor_model.utils.helper import create_train_test_df
from bike_sharing_regressor_model.data.preprocess import create_preprocessing_pipeline



def compare_between_models(df: pd.DataFrame) -> pd.DataFrame:

    X_train, X_test, y_train, y_test = create_train_test_df(df=df)

    rush_transformer, ct = create_preprocessing_pipeline()

    models = {
        'XGBRegressor': XGBRegressor(),
        'CatBoostRegressor': CatBoostRegressor(verbose=0, random_state=42),
        'LGBMRegressor': LGBMRegressor()
    }

    results = {}
    for model_name, model_obj in models.items():
        full_pipeline = Pipeline([
            ('rush_hrs', rush_transformer),
            ('preprocessing', ct),
            ('model', model_obj)
        ])
        full_pipeline.fit(X_train, y_train)
        y_pred_train = full_pipeline.predict(X_train)
        y_pred_test = full_pipeline.predict(X_test)
        results[model_name] = {
            "r2_train": r2_score(y_train, y_pred_train),
            "r2_test": r2_score(y_test, y_pred_test),
            "rmse_train": root_mean_squared_error(y_train, y_pred_train),
            "rmse_test": root_mean_squared_error(y_test, y_pred_test),
            "mae_train": mean_absolute_error(y_train, y_pred_train),
            "mae_test": mean_absolute_error(y_test, y_pred_test)
        }

    return pd.DataFrame(results)
