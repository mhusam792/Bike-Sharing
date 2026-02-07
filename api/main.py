from fastapi import FastAPI, status

from api.schemas import BikeSharingFeatures
from bike_sharing_model.models.predictor import predict_new_data_json
from bike_sharing_model.models.trainer import run_training

from typing import List

app = FastAPI()


@app.post(
    "/training",
    # response_model=TrainingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def train():
    return run_training(end_point=True, show_accuracy=True)


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(payload: List[BikeSharingFeatures]):
    json_data = [row.model_dump() for row in payload]

    result = predict_new_data_json(json_data)
    return result
