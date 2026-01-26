import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RushHourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: list[str], target: str = "cnt", top_n: int = 5):
        self.variables = variables
        self.target = target
        self.top_n = top_n
        self.top_hours_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        if y is None:
            raise ValueError("y must be provided for RushHourTransformer fit")

        for var in self.variables:
            temp_df = pd.DataFrame({var: X[var], self.target: y})
            top_hours = (
                temp_df.groupby(var)[self.target]
                .sum()
                .sort_values(ascending=False)
                .head(self.top_n)
                .index.tolist()
            )
            self.top_hours_[var] = top_hours

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_transformed = X.copy()
        for var in self.variables:
            X_transformed[f"{var}_rush_hr"] = (
                X_transformed[var].isin(self.top_hours_[var]).astype(int)
            )
        return X_transformed
