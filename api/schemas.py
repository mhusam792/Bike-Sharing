from typing import Dict

from pydantic import BaseModel


class ScoreModel(BaseModel):
    r2: float
    rmse: float
    mae: float


class ModelComparison(BaseModel):
    test_score: ScoreModel
    train_score: ScoreModel


class TrainingResponse(BaseModel):
    model_accuracy: Dict[str, ModelComparison]
    best_model_info: Dict[str, str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_accuracy": {
                    "CatBoostRegressor": {
                        "test_score": {
                            "r2": 0.9737885408445598,
                            "rmse": 27.029031558563954,
                            "mae": 17.930544633395183,
                        },
                        "train_score": {
                            "r2": 0.9198375595779287,
                            "rmse": 62.425805334940804,
                            "mae": 41.26206599226589,
                        },
                    },
                },
                "best_model_info": {
                    "test_csv_path": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/datasets/processed/test_split.csv",
                    "saved_model_path": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/models/catboost_regressor_model_output_v.pkl",
                },
            }
        }
    }


class PredictionResponse(BaseModel):
    pred_result_info: Dict[str, str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "pred_result_info": {
                    "trained_model": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/models/catboost_regressor_model_output_v.pkl",
                    "prediction_csv_path": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/outputs/predicted_results.csv",
                }
            }
        }
    }
