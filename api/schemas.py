from typing import Dict

from pydantic import BaseModel


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
