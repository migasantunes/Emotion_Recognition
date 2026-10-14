import numpy as np

# X: frames x coordinates, statistics computed per column (one coordinate at a time)
# returns a boolean matrix with the same shape, True = flagged

# Tukey's fences
def iqr_flags(X, k=1.5):
    X = np.asarray(X, dtype=float) # accepts a DataFrame too

    q1, q3 = np.percentile(X, [25, 75], axis=0) # one Q1 and one Q3 per column
    iqr = q3 - q1

    return (X < q1 - k * iqr) | (X > q3 + k * iqr) # broadcasting the fences over every frame

#z-score
# |z| > k, with z = (x - mean) / std; constant columns (std = 0) get z = 0
def zscore_flags(X, k):
    X = np.asarray(X, dtype=float)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0) # ddof=0
    # divide only where sigma > 0, the other cells keep the 0 from out
    z = np.divide(X - mu, sigma, out=np.zeros_like(X), where=sigma > 0)

    return np.abs(z) > k
