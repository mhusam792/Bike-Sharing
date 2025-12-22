from bike_sharing_regressor_model.models.train import create_best_model
from bike_sharing_regressor_model.models.evaluate import compare_between_models
from bike_sharing_regressor_model.config.settings import DATA_CONFIG
from bike_sharing_regressor_model.data.load import load_dataframe

TRAINING_DATA_FILE_PATH = f"datasets/{DATA_CONFIG['training_data_file']}"

df = load_dataframe(path=TRAINING_DATA_FILE_PATH)
results_df = compare_between_models(df)
print(results_df)
create_best_model(df)
