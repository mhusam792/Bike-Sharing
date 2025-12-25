from bike_sharing_model.config.settings import TESTING_DATA_FILE_PATH
from bike_sharing_model.data.loader import load_dataframe
from bike_sharing_model.models.predictor import predict_new_data


def run_prediction():
    X_new = load_dataframe(path=TESTING_DATA_FILE_PATH)
    predict_new_data(X_new)

if __name__ == "__main__":
    run_prediction()