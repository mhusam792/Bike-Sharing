from bike_sharing_model.models.trainer import create_best_model
from bike_sharing_model.models.evaluator import compare_between_models
from bike_sharing_model.config.core import TRAINING_DATA_FILE_PATH
from bike_sharing_model.data.loader import load_dataframe

def run_training():
    df = load_dataframe(path=TRAINING_DATA_FILE_PATH)
    results_df = compare_between_models(df)
    print(results_df)
    create_best_model(df)

if __name__ == "__main__":
    run_training()
