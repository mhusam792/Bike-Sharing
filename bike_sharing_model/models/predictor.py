import pandas as pd
import mlflow

from bike_sharing_model.config.core import (
    PREDICTION_PATH_FILE,
    TESTING_DATA_FILE_PATH,
)
from bike_sharing_model.data.loader import load_dataframe


def predict_new_data(
    X_new: pd.DataFrame,
    model_name: str = "bike_sharing_demand_model",
    model_stage: str = "Production",
    save_path: str = PREDICTION_PATH_FILE,
) -> dict:

    result = {}

    # Correct model URI
    model_uri = f"models:/{model_name}/{model_stage}"
    result["registered_model_uri"] = model_uri

    # Load model from MLflow Registry
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.sklearn.load_model(model_uri)

    # Predict
    y_pred = model.predict(X_new)
    X_new["predicted_cnt"] = y_pred

    # Save predictions
    X_new.to_csv(save_path, index=False)
    result["prediction_csv_path"] = str(save_path)

    return result


def run_prediction(end_point: bool = False) -> dict | None:
    X_new = load_dataframe(path=TESTING_DATA_FILE_PATH)
    pred_result = predict_new_data(X_new)

    if end_point:
        return {"pred_result_info": pred_result}
    print(pd.DataFrame.from_dict(pred_result, orient="index"))
    return None
