import pandas as pd
import numpy as np

X_COLS = [f"x_{i}" for i in range(68)]
Y_COLS = [f"y_{i}" for i in range(68)]

# x′k = (xk − x27) / d
# y′k = (yk − y27) / d
# d = sqrt((x45 − x36)^2 + (y45 − y36)^2)

def normalise_d(df):
    df_n = df.copy()

    x = df[X_COLS].to_numpy(dtype=float)
    y = df[Y_COLS].to_numpy(dtype=float)
    d = np.sqrt((x[:, 45] - x[:, 36]) ** 2 + (y[:, 45] - y[:, 36]) ** 2)
    d[d == 0] = np.nan

    df_n[X_COLS] = (x - x[:, [27]]) / d[:, None] # broadcasting d
    df_n[Y_COLS] = (y - y[:, [27]]) / d[:, None] # same here

    return df_n.copy()

# x′k = (xk − x27) / w
# y′k = (yk − y27) / h

def normalise_image(df, w=1280, h=720):
    df_n = df.copy()

    x = df[X_COLS].to_numpy(dtype=float)
    y = df[Y_COLS].to_numpy(dtype=float)

    df_n[X_COLS] = (x - x[:, [27]]) / w
    df_n[Y_COLS] = (y - y[:, [27]]) / h

    return df_n.copy()