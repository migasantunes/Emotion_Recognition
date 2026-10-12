import pandas as pd
import numpy as np

def euclidean_distance(df, a, b):
    x_a = df[f"x_{a}"].to_numpy(dtype=float)
    y_a = df[f"y_{a}"].to_numpy(dtype=float)
    x_b = df[f"x_{b}"].to_numpy(dtype=float)
    y_b = df[f"y_{b}"].to_numpy(dtype=float)

    return pd.Series(np.sqrt(np.square(x_a - x_b) + np.square(y_a - y_b)), index=df.index)