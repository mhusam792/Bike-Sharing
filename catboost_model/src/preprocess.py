import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from feature_engine.creation import CyclicalFeatures
from functools import partial

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
    cyclical_cols = ['season', 'mnth', 'weekday', 'hr']
    new_feat = ['casual', 'registered']
    num_cols = ['temp', 'atemp', 'hum', 'windspeed']
    cat_cols = ['weathersit']

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
        ('cyclical', cyclical_pipe, cyclical_cols),
        ('scaling', numeric_pipe, num_cols),
        ('ohe', categories_pipe, cat_cols)
    ], remainder='passthrough').set_output(transform='pandas')

    rush_transformer = FunctionTransformer(
        func=partial(get_rush_hours, variables=new_feat),
        validate=False
    )

    return rush_transformer, ct
