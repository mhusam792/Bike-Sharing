import pandas as pd

from sklearn.pipeline import Pipeline
import joblib

from catboost import CatBoostRegressor
                                           
from bike_sharing_model.config.settings import TRAINED_MODEL_PATH, TESTING_DATA_FILE_PATH
from bike_sharing_model.utils.helpers import create_train_test_df
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline

                                                           

def create_best_model(df: pd.DataFrame, save_path=TRAINED_MODEL_PATH):
    X_train, X_test, y_train, y_test = create_train_test_df(df=df)

    test_df = X_test.copy()
    test_df["cnt"] = y_test.values
    test_df.to_csv(TESTING_DATA_FILE_PATH, index=False)
    print(f"Test split saved to {TESTING_DATA_FILE_PATH}")

    rush_transformer, ct = create_preprocessing_pipeline()

    best_model_pipeline = Pipeline([
        ('rush_hrs', rush_transformer),
        ('preprocessing', ct),
        ('model', CatBoostRegressor(verbose=0, random_state=42))
    ])
    best_model_pipeline.fit(X_train, y_train)
    joblib.dump(best_model_pipeline, f"{save_path}")
    print(f"Best model saved to {save_path}")
