import pandas as pd
import joblib
from bike_sharing_model.config.core import (TRAINED_MODEL_PATH, 
                                                PREDICTION_PATH_FILE)


def predict_new_data(X_new: pd.DataFrame, 
                     model_path=TRAINED_MODEL_PATH, 
                     save_path=PREDICTION_PATH_FILE) -> None:
    
    print(f"Pipeline model: {TRAINED_MODEL_PATH}")

    model_pipeline = joblib.load(model_path)
    y_pred = model_pipeline.predict(X_new)
    X_new['predicted_cnt'] = y_pred
    X_new.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")
