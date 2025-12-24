from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional


class SplitConfig(BaseModel):
    test_size: float
    random_state: int

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v):
        if not 0 < v < 1:
            raise ValueError("test_size must be between 0 and 1")
        return v

class DataConfig(BaseModel):
    # Package
    package_name: str

    # Files
    training_data_file: str
    test_data_file: str
    trained_model_path: Optional[str] = None

    # Target & pipeline
    target: str
    pipeline_name: str
    pipeline_save_file: str

    # Features
    features: List[str]
    cyclical_cols: List[str]
    num_cols: List[str]
    cat_cols: List[str]
    new_features: List[str]

    # Split
    split: SplitConfig

    @model_validator(mode="after")
    def validate_columns(self):
        all_features = set(self.features)

        # target must not be in features
        if self.target in all_features:
            raise ValueError("target column must NOT be inside features")

        # cyclical ⊆ features
        if not set(self.cyclical_cols).issubset(all_features):
            raise ValueError("All cyclical_cols must be in features")

        # numerical ⊆ features
        if not set(self.num_cols).issubset(all_features):
            raise ValueError("All num_cols must be in features")

        # categorical ⊆ features
        if not set(self.cat_cols).issubset(all_features):
            raise ValueError("All cat_cols must be in features")

        # new_features ⊆ features
        if not set(self.new_features).issubset(all_features):
            raise ValueError("All new_features must be in features")

        return self


import yaml
from pathlib import Path


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


config_dict = load_yaml(Path("configs/data.yml"))

data_config = DataConfig(**config_dict)

print("✅ Data config validated successfully")
