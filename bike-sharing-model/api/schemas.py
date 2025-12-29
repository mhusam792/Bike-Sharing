from pydantic import BaseModel
from typing import Dict, List

class ScoreModel(BaseModel):
    r2: float
    rmse: float
    mae: float

class ModelComparison(BaseModel):
    test_score: ScoreModel
    train_score: ScoreModel

class TrainingResponse(BaseModel):
    comparing_models: Dict[str, ModelComparison]
    best_model_info: Dict[str, str]
    
    model_config = {
        'json_schema_extra':{
            'example': {
                "comparing_models": {
                    "XGBRegressor": {
                    "test_score": {
                        "r2": 0.9698052406311035,
                        "rmse": 29.01019287109375,
                        "mae": 19.430381774902344
                    },
                    "train_score": {
                        "r2": 0.8900954723358154,
                        "rmse": 73.09478759765625,
                        "mae": 49.23468780517578
                    }
                    },
                    "CatBoostRegressor": {
                    "test_score": {
                        "r2": 0.9737885408445598,
                        "rmse": 27.029031558563954,
                        "mae": 17.930544633395183
                    },
                    "train_score": {
                        "r2": 0.9198375595779287,
                        "rmse": 62.425805334940804,
                        "mae": 41.26206599226589
                    }
                    },
                    "LGBMRegressor": {
                    "test_score": {
                        "r2": 0.9762889999318385,
                        "rmse": 25.7074986777784,
                        "mae": 17.043272842683464
                    },
                    "train_score": {
                        "r2": 0.912388637407682,
                        "rmse": 65.26177867716812,
                        "mae": 43.20901531632474
                    }
                    }
                },
                "best_model_info": {
                    "test_csv_path": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/datasets/processed/test_split.csv",
                    "saved_model_path": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/models/catboost_regressor_model_output_v.pkl"
                }
                }
        }
    }

class PredictionResponse(BaseModel):
    pred_result_info: Dict[str, str]

    model_config = {
        'json_schema_extra':{
            'example':{
                "pred_result_info": {
                    "trained_model": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/models/catboost_regressor_model_output_v.pkl",
                    "prediction_csv_path": "/home/mohamedhussam/ds_projects/portofolio/bike_sharing/bike-sharing-model/outputs/predicted_results.csv"
                    }
                }
        }
    }
    