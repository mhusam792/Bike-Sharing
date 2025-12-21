import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from feature_engine.creation import CyclicalFeatures
from functools import partial
from bike_sharing_regressor_model.config.settings import DATA_CONFIG

CYCLICAL_FEATURES = DATA_CONFIG['cyclical_cols']
NUMERICAL_FEATURES = DATA_CONFIG['num_cols']
CATEGORICAL_FEAUTRES = DATA_CONFIG['cat_cols']
NEW_FEATURES = DATA_CONFIG['new_features']

def get_rush_hours(X: pd.DataFrame, variables: list[str]):
    X = X.copy()
    for col in variables:
        rush_hrs = (
            X.groupby('hr')[col].sum()
             .sort_values(ascending=False)
             .head(5)
             .index.tolist()
        )
        X[f"{col}_rush_hrs"] = X['hr'].isin(rush_hrs).astype(int)
        X.drop(columns=[col], inplace=True)
    return X

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

    ct = ColumnTransformer(transformers=[
        ('cyclical', cyclical_pipe, CYCLICAL_FEATURES),
        ('scaling', numeric_pipe, NUMERICAL_FEATURES),
        ('ohe', categories_pipe, CATEGORICAL_FEAUTRES)
    ], remainder='passthrough').set_output(transform='pandas')

    rush_transformer = FunctionTransformer(
        func=partial(get_rush_hours, variables=NEW_FEATURES),
        validate=False
    )

    return rush_transformer, ct
