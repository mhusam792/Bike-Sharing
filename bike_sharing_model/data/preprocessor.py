from feature_engine.creation import CyclicalFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_sharing_model.config.core import DATA_CONFIG

CYCLICAL_FEATURES = DATA_CONFIG.cyclical_cols
NUMERICAL_FEATURES = DATA_CONFIG.num_cols
CATEGORICAL_FEATURES = DATA_CONFIG.cat_cols
NEW_FEATURES = DATA_CONFIG.new_features


def create_preprocessing_pipeline():

    cyclical_pipe = Pipeline(
        [("cyclical_transformation", CyclicalFeatures(drop_original=True))]
    )
    numeric_pipe = Pipeline([("scaling_numbers", StandardScaler())])
    categories_pipe = Pipeline([("ohe", OneHotEncoder(sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cyclical", cyclical_pipe, CYCLICAL_FEATURES),
            ("scaling", numeric_pipe, NUMERICAL_FEATURES),
            ("ohe", categories_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
