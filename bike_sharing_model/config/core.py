from importlib import resources
from pathlib import Path

import yaml

from bike_sharing_model.data.yml_validator import ValidateInputs


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_config_from_package() -> dict:
    with resources.open_text("bike_sharing_model.configs", "data.yml") as f:
        return yaml.safe_load(f)


DATA_CONFIG = ValidateInputs(**load_config_from_package())

# Reading Train, Test Datafiles and save predictions to specific path
TRAINING_DATA_FILE_PATH = DATA_CONFIG.training_data_path
TESTING_DATA_FILE_PATH = DATA_CONFIG.test_data_path

# Features and Target
FEATURES_LIST = DATA_CONFIG.features
TARGET = DATA_CONFIG.target

# Train Test split config
TEST_SIZE = DATA_CONFIG.split.test_size
RANDOM_STATE = DATA_CONFIG.split.random_state
