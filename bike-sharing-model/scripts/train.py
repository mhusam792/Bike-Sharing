from bike_sharing_model.models.trainer import create_best_model
from bike_sharing_model.models.evaluator import compare_between_models
from bike_sharing_model.config.core import TRAINING_DATA_FILE_PATH
from bike_sharing_model.data.loader import load_dataframe

import pandas as pd

def run_training():
    df = load_dataframe(path=TRAINING_DATA_FILE_PATH)

    comparing_models = compare_between_models(df)
    print(pd.DataFrame(comparing_models))

    best_model_info = create_best_model(df)
    print(pd.DataFrame.from_dict(best_model_info, orient="index"))


if __name__ == "__main__":
    run_training()
