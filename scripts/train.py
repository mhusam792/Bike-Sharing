# from bike_sharing_model.models.trainer import run_training
from bike_sharing_model.mlflow_runner import run_training_with_mlflow


if __name__ == "__main__":
    run_training_with_mlflow()
