from typing import List, Optional

from pydantic import BaseModel, Field


class Split(BaseModel):
    test_size: float = Field(gt=0, lt=1)
    random_state: int = Field(ge=0, le=4294967295)


class ValidateInputs(BaseModel):
    # Package Overview
    package_name: str

    # Package Overview
    training_data_file: str
    test_data_file: str

    ## Variables / Features
    # The variable we are attempting to predict (cnt)
    target: str

    features: List[str]

    # Cyclical Features
    cyclical_cols: List[str]

    # Numerical Features
    num_cols: List[str]

    # Categorical Features
    cat_cols: List[str]

    # Make new features from existing features
    new_features: Optional[List[str]] = None

    # Set train/test split
    split: Split
