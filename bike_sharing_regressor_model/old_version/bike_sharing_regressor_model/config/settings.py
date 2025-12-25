# settings.py
import yaml
import bike_sharing_regressor_model
from pathlib import Path


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

# Package and Root Path
PACKAGE_ROOT = Path(bike_sharing_regressor_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent

# Reading Configrations from data.yml file
DATA_CONFIG_PATH = ROOT / "configs" / "data.yml"
DATA_CONFIG = load_config(DATA_CONFIG_PATH)

# Directory Paths for Datasets, Predictions and Trained Models
DATASET_DIR = ROOT / "datasets"
PREDICTION_DIR = ROOT / 'predictions'
TRAINED_MODEL_DIR = ROOT / "trained_models"

# Reading Train, Test Datafiles and save predictions to specific path
TRAINING_DATA_FILE_PATH = DATASET_DIR / DATA_CONFIG['training_data_file']
TESTING_DATA_FILE_PATH = DATASET_DIR / DATA_CONFIG['test_data_file']
PREDICTION_PATH_FILE = PREDICTION_DIR / 'predicted_results.csv'

# Trained Model Path
TRAINED_MODEL_PATH = (
    TRAINED_MODEL_DIR / f"{DATA_CONFIG['pipeline_save_file']}.pkl"
)

# Features and Target
FEATURES_LIST = DATA_CONFIG['features']
TARGET = DATA_CONFIG['target']

# Train Test split config
TEST_SIZE = DATA_CONFIG['split']['test_size']
RANDOM_STATE = DATA_CONFIG['split']['random_state']
