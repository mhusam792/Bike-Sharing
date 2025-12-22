import pandas as pd


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
