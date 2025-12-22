import pandas as pd

from sklearn.pipeline import Pipeline
import joblib

from catboost import CatBoostRegressor
                                           
from bike_sharing_regressor_model.config.settings import TRAINED_MODEL_PATH
from bike_sharing_regressor_model.utils.helper import create_train_test_df
from bike_sharing_regressor_model.data.preprocess import create_preprocessing_pipeline

                                                           

def create_best_model(df: pd.DataFrame, save_path=TRAINED_MODEL_PATH):
    X_train, X_test, y_train, y_test = create_train_test_df(df=df)

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
    joblib.dump(best_model_pipeline, f"{save_path}")
    print(f"Best model saved to {save_path}.pkl")
