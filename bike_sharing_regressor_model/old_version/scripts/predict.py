from bike_sharing_regressor_model.config.settings import DATA_CONFIG
from bike_sharing_regressor_model.data.load import load_dataframe
from bike_sharing_regressor_model.models.predict import predict_new_data

TESTING_DATA_FILE_PATH = f"datasets/{DATA_CONFIG['training_data_file']}"

def run_prediction():
    X_new = load_dataframe(path=TESTING_DATA_FILE_PATH)
    predict_new_data(X_new)

if __name__ == "__main__":
    run_prediction()