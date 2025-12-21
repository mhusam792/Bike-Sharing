from bike_sharing_regressor_model.utils.helper import load_config
import bike_sharing_regressor_model
from pathlib import Path


PACKAGE_ROOT = Path(bike_sharing_regressor_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent

DATA_CONFIG_PATH = ROOT / "configs" / "data.yml"
DATA_CONFIG = load_config(DATA_CONFIG_PATH)

TRAINING_DATA_FILE_PATH = ROOT / "datasets" / DATA_CONFIG['training_data_file']
TESTING_DATA_FILE_PATH = ROOT / "datasets" / DATA_CONFIG['test_data_file']

DATASET_DIR = ROOT / "datasets"
TRAINED_MODEL_PATH = ROOT / "trained_models" / DATA_CONFIG['pipeline_save_file']

FEATURES_LIST = DATA_CONFIG['features']
TARGET = DATA_CONFIG['target']

TEST_SIZE = DATA_CONFIG['split']['test_size']
RANDOM_STATE = DATA_CONFIG['split']['random_state']
