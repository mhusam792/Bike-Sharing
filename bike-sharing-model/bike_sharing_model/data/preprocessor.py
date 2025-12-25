from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from feature_engine.creation import CyclicalFeatures

from functools import partial

from bike_sharing_model.config.settings import DATA_CONFIG
from bike_sharing_model.features.feature_engineering import get_rush_hours

CYCLICAL_FEATURES = DATA_CONFIG['cyclical_cols']
NUMERICAL_FEATURES = DATA_CONFIG['num_cols']
CATEGORICAL_FEAUTRES = DATA_CONFIG['cat_cols']
NEW_FEATURES = DATA_CONFIG['new_features']

def create_preprocessing_pipeline():

    cyclical_pipe = Pipeline([
        ('cyclical_transformation', CyclicalFeatures(drop_original=True))
    ])
    numeric_pipe = Pipeline([
        ('scaling_numbers', StandardScaler())
    ])
    categories_pipe = Pipeline([
        ('ohe', OneHotEncoder(sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('cyclical', cyclical_pipe, CYCLICAL_FEATURES),
        ('scaling', numeric_pipe, NUMERICAL_FEATURES),
        ('ohe', categories_pipe, CATEGORICAL_FEAUTRES)
    ], remainder='passthrough').set_output(transform='pandas')

    rush_transformer = FunctionTransformer(
        func=partial(get_rush_hours, variables=NEW_FEATURES),
        validate=False
    )

    return rush_transformer, preprocessor
