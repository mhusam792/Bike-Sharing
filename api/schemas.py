from typing import Dict

from pydantic import BaseModel, Field


class Score(BaseModel):
    r2: float
    rmse: float
    mae: float


class ModelScore(BaseModel):
    test_score: Score
    train_score: Score


class BestModelInfo(BaseModel):
    test_csv_path: str
    saved_model_path: str


class TrainingResponse(BaseModel):
    model_accuracy: Dict[str, ModelScore]
    best_model_info: BestModelInfo


class PredResultInfo(BaseModel):
    trained_model: str
    prediction_csv_path: str


class PredictionResponse(BaseModel):
    pred_result_info: PredResultInfo


class BikeSharingFeatures(BaseModel):
    season: int
    yr: int = Field(ge=0, le=1)
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=1)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temp: float
    atemp: float
    hum: float
    windspeed: float
