"""The group's pipeline, called by the application (app.py).

This is the only file of the application to be changed. Implement two functions:

    load_model()            -> called once, when the application starts
    predict(model, frames)  -> called for every recording

`frames` is a pandas DataFrame in the format of the CSV files of the dataset, one row
per frame at 30 frames per second, with the columns

    frame, timestamp, confidence, x_0 ... x_67, y_0 ... y_67

in pixels (origin at the top-left corner, y increasing downwards). The coordinates are
NOT normalised: apply the same normalisation and compute the features with the same
code that was used for training, imported from your own modules, not rewritten here.

`predict` returns a dict with one probability per emotion, in the order of EMOTIONS.
"""
import os
import joblib
import numpy as np
import pandas as pd

from src.normalise import normalise_d
from src.features import video_features

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")


def load_model():
    """Load the pipeline of 5.6, fitted on the 1440 videos in 06_deployment_7_Answer.ipynb."""
    return joblib.load(MODEL_PATH)


def predict(model, frames: pd.DataFrame) -> dict:
    """Return {emotion: probability} for one recording."""
    frames_n = normalise_d(frames)                          # pixels -> units of d, as in training
    features = video_features(frames_n).to_frame().T        # one row, the 55 features
    probabilities = model.predict_proba(features)[0]

    # model.classes_ are the emotion codes 1..8
    result = {}
    for code, p in zip(model.classes_, probabilities):
        result[EMOTIONS[code - 1]] = p
    return result