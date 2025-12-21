# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from bike_sharing_regressor_model.models.preprocess import create_preprocessing_pipeline
import joblib
from bike_sharing_regressor_model.config.settings import DATA_CONFIG
from typing import List, Tuple


def _create_train_test_df(
        df: pd.DataFrame, 
        features: List[str] | None=None, 
        target: str | None =None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    if features is None or target is None:
        features = DATA_CONFIG['features']
        target = DATA_CONFIG['target']

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=DATA_CONFIG['split']['test_size'], 
        shuffle=False, 
        random_state=DATA_CONFIG['split']['random_state']
    )

    return X_train, X_test, y_train, y_test

def compare_between_models(df: pd.DataFrame) -> pd.DataFrame:

    X_train, X_test, y_train, y_test = _create_train_test_df(df=df)

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

def create_best_model(df: pd.DataFrame, save_path='trained_models/catboost_pipeline.pkl'):
    X_train, X_test, y_train, y_test = _create_train_test_df(df=df)

    test_df = X_test.copy()
    test_df["cnt"] = y_test.values
    test_df.to_csv("datasets/test_split.csv", index=False)
    print("Test split saved to data/test_split.csv")

    rush_transformer, ct = create_preprocessing_pipeline()

    best_model_pipeline = Pipeline([
        ('rush_hrs', rush_transformer),
        ('preprocessing', ct),
        ('model', CatBoostRegressor(verbose=0, random_state=42))
    ])
    best_model_pipeline.fit(X_train, y_train)
    joblib.dump(best_model_pipeline, save_path)
    print(f"Best model saved to {save_path}")
