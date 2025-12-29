import pandas as pd
import joblib
from bike_sharing_model.config.core import (TESTING_DATA_FILE_PATH, 
                                            TRAINED_MODEL_PATH, 
                                            PREDICTION_PATH_FILE)

from bike_sharing_model.data.loader import load_dataframe


def predict_new_data(X_new: pd.DataFrame, 
                     model_path=TRAINED_MODEL_PATH, 
                     save_path=PREDICTION_PATH_FILE) -> dict:
    
    result = dict()
    
    result['trained_model'] = str(TRAINED_MODEL_PATH)

    model_pipeline = joblib.load(model_path)
    y_pred = model_pipeline.predict(X_new)
    X_new['predicted_cnt'] = y_pred

    X_new.to_csv(save_path, index=False)
    result['prediction_csv_path'] = str(save_path)

    return result

def run_prediction(end_point:bool=False) -> dict|None:
    X_new = load_dataframe(path=TESTING_DATA_FILE_PATH)
    pred_result = predict_new_data(X_new)

    if end_point:
        return {
            'pred_result_info': pred_result
        }
    print(pd.DataFrame.from_dict(pred_result, orient="index"))