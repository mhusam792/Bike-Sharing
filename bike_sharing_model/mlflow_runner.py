import mlflow
from bike_sharing_model.models.trainer import run_training


def run_training_with_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("bike-sharing-training")

    with mlflow.start_run(run_name="catboost_pipeline"):

        result = run_training(end_point=True, show_accuracy=True)

        # ===== metrics =====
        model_acc = result["model_accuracy"]
        for model_name, scores in model_acc.items():
            for split_name, metrics in scores.items():
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(
                        f"{model_name}_{split_name}_{metric_name}", float(metric_value)
                    )

        # ===== artifacts =====
        best_model_info = result["best_model_info"]
        mlflow.log_artifact(best_model_info["test_csv_path"])

        # ===== params =====
        mlflow.log_params(
            {
                "model": "CatBoostRegressor",
                "random_state": 42,
                "rush_hour_top_n": 5,
                "time_series": True,
                "shuffle": False,
                "target": "cnt",
            }
        )

        return result
