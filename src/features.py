import pandas as pd
import numpy as np

def euclidean_distance(df, a, b):
    x_a = df[f"x_{a}"].to_numpy(dtype=float)
    y_a = df[f"y_{a}"].to_numpy(dtype=float)
    x_b = df[f"x_{b}"].to_numpy(dtype=float)
    y_b = df[f"y_{b}"].to_numpy(dtype=float)

    return pd.Series(np.sqrt(np.square(x_a - x_b) + np.square(y_a - y_b)), index=df.index)

"""
Before this measures method version, I had one using numpy with one loop per eye and brow,
passing the collumns to a numpy float array and calculating the mean of the 6 or 5 points.
It was a lot faster than the current one, around 5 times faster, but it wasn't as readable and 
was using loops, which is not the best practice in pandas. So I decided to change it to this version.
"""
def measures(df):
    mouth_vertical_mean = df[["y_51", "y_57"]].mean(axis=1)
    mouth_corner_mean = df[["y_48", "y_54"]].mean(axis=1)

    M3 = mouth_vertical_mean - mouth_corner_mean

    left_eye_height = df[[f"y_{36 + i}" for i in range(6)]].mean(axis=1) # Actor's Right Eye
    right_eye_height = df[[f"y_{42 + i}" for i in range(6)]].mean(axis=1) # Actor's Left Eye
    left_brow_height = df[[f"y_{17 + i}" for i in range(5)]].mean(axis=1) # Actor's Right Brow
    right_brow_height = df[[f"y_{22 + i}" for i in range(5)]].mean(axis=1) # Actor's Left Brow

    M4 = (left_eye_height - left_brow_height + 
          right_eye_height - right_brow_height) / 2
    
    M5 = (euclidean_distance(df, 37, 41) + 
          euclidean_distance(df, 38, 40) + 
          euclidean_distance(df, 43, 47) + 
          euclidean_distance(df, 44, 46)) / 4

    return pd.DataFrame({"M1": euclidean_distance(df, 62, 66), 
                         "M2": euclidean_distance(df, 48, 54),
                         "M3": M3,
                         "M4": M4,
                         "M5": M5, 
                         "M6": euclidean_distance(df, 21, 22)})

def own_measures(df):
    mouth_eye_left_corners = euclidean_distance(df, 48, 36) # Actor's right
    mouth_eye_right_corners = euclidean_distance(df, 54, 45) # Actor's left
    M7 = (mouth_eye_left_corners + mouth_eye_right_corners) / 2

    lower_lip = df[[f"y_{55 + i}" for i in range(5)]].mean(axis=1)
    lower_nose = df[[f"y_{31 + i}" for i in range(5)]].mean(axis=1)
    M8 = lower_lip - lower_nose

    inner_ends_brow_height = df[["y_21", "y_22"]].mean(axis=1)
    outer_ends_brow_height = df[["y_17", "y_26"]].mean(axis=1)
    M9 = inner_ends_brow_height - outer_ends_brow_height

    higher_lip = df[[f"y_{49 + i}" for i in range(5)]].mean(axis=1)
    M10 = higher_lip - lower_nose

    left_brow_arch = df[["y_17", "y_21"]].mean(axis=1) - df["y_19"]
    right_brow_arch = df[["y_22", "y_26"]].mean(axis=1) - df["y_24"]
    M11 = (left_brow_arch + right_brow_arch) / 2

    return pd.DataFrame({"own_M7": M7,
                         "own_M8": M8,
                         "own_M9": M9,
                         "own_M10": M10,
                         "own_M11": M11})

def video_features(video_df, own):
    if own:
        measures_df = pd.concat([measures(video_df), own_measures(video_df)], axis=1)
    else:
        measures_df = measures(video_df)

    final_measures = pd.DataFrame({"mean": measures_df.mean(),
                                   "std": measures_df.std(ddof=0),
                                   "range": measures_df.max() - measures_df.min(),
                                   "vel": measures_df.diff().abs().mean(),
                                   "energy": measures_df.diff().pow(2).mean()})

    final_measures = final_measures.stack()
    final_measures.index = final_measures.index.map("_".join)

    return final_measures

def build_feature_table(df, own=True):
    return df.groupby(["video_id", "actor", "emotion"]).apply(video_features, own=own)