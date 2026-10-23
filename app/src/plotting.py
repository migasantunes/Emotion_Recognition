import numpy as np
import matplotlib.pyplot as plt

# Facial regions of Figure 2: (first landmark, last landmark, closed contour)
REGIONS = {
    "jaw": (0, 16, False),
    "right_brow": (17, 21, False),
    "left_brow": (22, 26, False),
    "nose_bridge": (27, 30, False),
    "nose_base": (31, 35, False),
    "right_eye": (36, 41, True),
    "left_eye": (42, 47, True),
    "outer_lip": (48, 59, True),
    "inner_lip": (60, 67, True),
}


def frame_xy(frame):
    # The 68 x and 68 y coordinates of one row of the frame table
    x = np.array([frame[f"x_{k}"] for k in range(68)])
    y = np.array([frame[f"y_{k}"] for k in range(68)])
    return x, y


def draw_landmarks(frame, ax=None, color="tab:blue", alpha=1.0, points=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    x, y = frame_xy(frame)

    for first, last, closed in REGIONS.values():
        idx = list(range(first, last + 1))
        if closed:
            idx.append(first)
        ax.plot(x[idx], y[idx], color=color, alpha=alpha, linewidth=1)

    if points:
        ax.scatter(x, y, color=color, alpha=alpha, s=8)

    # Image coordinates: y grows downwards, and both axes share the same scale
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_aspect("equal")
    return ax


def overlay_video(video_frames, ax=None, color="tab:blue", alpha=0.08):
    # All the frames of one video drawn on the same axes, almost transparent
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    for _, frame in video_frames.iterrows():
        draw_landmarks(frame, ax=ax, color=color, alpha=alpha, points=False)

    return ax
