import pandas as pd
import mlflow
import dagshub

import copy

from bike_sharing_model.config.core import FEATURES_LIST


def predict_new_data_json(
    json_data: list,
    model_name: str = "bike_sharing_demand_model",
    model_stage: str = "Production",
) -> list:

    data_copy = copy.deepcopy(json_data)

    X = pd.DataFrame(data_copy)[FEATURES_LIST]

    # Load model
    # mlflow.set_tracking_uri("http://localhost:5000")
    dagshub.init(repo_owner="Mohamed_Hussam", repo_name="Bike-Sharing", mlflow=True)
    model_uri = f"models:/{model_name}/{model_stage}"
    model = mlflow.sklearn.load_model(model_uri)

    # Predict
    y_pred = model.predict(X)

    for row, pred in zip(data_copy, y_pred):
        row["predicted_cnt"] = float(pred)

    return data_copy
