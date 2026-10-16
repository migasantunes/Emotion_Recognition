import pandas as pd
import numpy as np

def euclidean_distance(df, a, b):
    x_a = df[f"x_{a}"].to_numpy(dtype=float)
    y_a = df[f"y_{a}"].to_numpy(dtype=float)
    x_b = df[f"x_{b}"].to_numpy(dtype=float)
    y_b = df[f"y_{b}"].to_numpy(dtype=float)

    return pd.Series(np.sqrt(np.square(x_a - x_b) + np.square(y_a - y_b)), index=df.index)

def measures(df):
    mouth_vertical_mean = (df["y_51"].to_numpy(dtype=float) + df["y_57"].to_numpy(dtype=float)) / 2
    mouth_corner_mean = (df["y_48"].to_numpy(dtype=float) + df["y_54"].to_numpy(dtype=float)) / 2
    M3 = mouth_vertical_mean - mouth_corner_mean

    left_eye_height = np.zeros(df.shape[0]) # Actors Right Eye
    right_eye_height = np.zeros(df.shape[0]) # Actors Left Eye
    for i in range(6):
        left_eye_height += df[f"y_{36 + i}"].to_numpy(dtype=float)
        right_eye_height += df[f"y_{42 + i}"].to_numpy(dtype=float)

    left_brow_height = np.zeros(df.shape[0]) # Actors Right Brow
    right_brow_height = np.zeros(df.shape[0]) # Actors Left Brow
    for i in range(5):
        left_brow_height += df[f"y_{17 + i}"].to_numpy(dtype=float)
        right_brow_height += df[f"y_{22 + i}"].to_numpy(dtype=float)

    M4 = (left_eye_height / 6 - left_brow_height / 5 + 
          right_eye_height / 6 - right_brow_height / 5) / 2
    
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

def video_features(video_df):
    measures_df = measures(video_df)

    final_measures = pd.DataFrame({"mean": measures_df.mean(),
                                   "std": measures_df.std(ddof=0),
                                   "range": measures_df.max() - measures_df.min(),
                                   "vel": measures_df.diff().abs().mean(),
                                   "energy": measures_df.diff().pow(2).mean()})

    final_measures = final_measures.stack()
    final_measures.index = final_measures.index.map("_".join)

    final_measures = pd.concat([final_measures, own_video_features(video_df)], axis=1)

    return final_measures

def own_video_features(video_df): # If needed, add measures_df

    return pd.DataFrame({})

def build_feature_table(df):

    return df.groupby("video_id").apply(video_features)