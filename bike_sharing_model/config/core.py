from importlib import resources
from pathlib import Path

import yaml

import bike_sharing_model
from bike_sharing_model.data.yml_validator import ValidateInputs


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_config_from_package() -> dict:
    with resources.open_text("bike_sharing_model.configs", "data.yml") as f:
        return yaml.safe_load(f)


DATA_CONFIG = ValidateInputs(**load_config_from_package())


Base_DIR = Path.cwd()

# Directory Paths for Datasets, Predictions and Trained Models
DATASET_DIR = Base_DIR / "datasets"

PREDICTION_DIR = Base_DIR / "outputs"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_MODEL_DIR = Base_DIR / "models"
TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Reading Train, Test Datafiles and save predictions to specific path
TRAINING_DATA_FILE_PATH = DATASET_DIR / "raw" / DATA_CONFIG.training_data_file
TESTING_DATA_FILE_PATH = DATASET_DIR / "processed" / DATA_CONFIG.test_data_file
PREDICTION_PATH_FILE = PREDICTION_DIR / "predicted_results.csv"

# Trained Model Path
TRAINED_MODEL_PATH = TRAINED_MODEL_DIR / f"{DATA_CONFIG.pipeline_save_file}.pkl"

# Features and Target
FEATURES_LIST = DATA_CONFIG.features
TARGET = DATA_CONFIG.target

# Train Test split config
TEST_SIZE = DATA_CONFIG.split.test_size
RANDOM_STATE = DATA_CONFIG.split.random_state
