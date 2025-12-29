from fastapi import FastAPI
from starlette import status
from api.schemas import TrainingResponse, PredictionResponse

from bike_sharing_model.models.predictor import run_prediction
from bike_sharing_model.models.trainer import run_training


app = FastAPI()

@app.post('/new_training', 
          response_model=TrainingResponse,
          status_code=status.HTTP_201_CREATED)
async def train():
    return run_training(end_point=True)

@app.post('/predict', 
          response_model=PredictionResponse,
          status_code=status.HTTP_200_OK)
async def predict():
    return run_prediction(end_point=True)
