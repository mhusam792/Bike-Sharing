import pandas as pd

from sklearn.pipeline import Pipeline
import joblib

from catboost import CatBoostRegressor
                                           
from bike_sharing_model.config.core import (TRAINED_MODEL_PATH, 
                                            TESTING_DATA_FILE_PATH, 
                                            RANDOM_STATE, 
                                            TRAINING_DATA_FILE_PATH)

from bike_sharing_model.data.loader import load_dataframe
from bike_sharing_model.models.evaluator import compare_between_models
from bike_sharing_model.utils.helpers import create_train_test_df
from bike_sharing_model.data.preprocessor import create_preprocessing_pipeline
                                                

def create_best_model(df: pd.DataFrame, 
                      save_path=TRAINED_MODEL_PATH,
                      ) -> dict:
    
    result = dict()

    X_train, X_test, y_train, y_test = create_train_test_df(df=df)

    test_df = X_test.copy()
    test_df["cnt"] = y_test.values
    test_df.to_csv(TESTING_DATA_FILE_PATH, index=False)

    result['test_csv_path'] = str(TESTING_DATA_FILE_PATH)

    rush_transformer, ct = create_preprocessing_pipeline()

    best_model_pipeline = Pipeline([
        ('rush_hrs', rush_transformer),
        ('preprocessing', ct),
        ('model', CatBoostRegressor(verbose=0, random_state=RANDOM_STATE))
    ])

    best_model_pipeline.fit(X_train, y_train)

    joblib.dump(best_model_pipeline, f"{save_path}")
    result['saved_model_path'] = str(save_path)

    return result


def run_training(end_point:bool=False) -> dict|None:
    df = load_dataframe(path=TRAINING_DATA_FILE_PATH)

    comparing_models = compare_between_models(df)
    best_model_info = create_best_model(df)

    if end_point:
        return {
            'comparing_models': comparing_models,
            'best_model_info': best_model_info
        }
    print(pd.DataFrame(comparing_models))
    print(pd.DataFrame.from_dict(best_model_info, orient="index"))
