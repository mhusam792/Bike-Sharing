from fastapi import FastAPI
from starlette import status

from api.schemas import PredictionResponse, TrainingResponse
from bike_sharing_model.models.predictor import run_prediction
from bike_sharing_model.models.trainer import run_training

app = FastAPI()


@app.post(
    "/training",
    # response_model=TrainingResponse,
    status_code=status.HTTP_201_CREATED,
)
async def train():
    return run_training(end_point=True, show_accuracy=True)


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict():
    return run_prediction(end_point=True)
