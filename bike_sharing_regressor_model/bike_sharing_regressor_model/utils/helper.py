
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import List, Tuple

from bike_sharing_regressor_model.config.settings import (
    FEATURES_LIST,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE
)





def create_train_test_df(
        df: pd.DataFrame, 
        features: List[str] | None=None, 
        target: str | None =None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    if features is None or target is None:
        features = FEATURES_LIST
        target = TARGET

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=TEST_SIZE, 
        shuffle=False, 
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test
