import numpy as np
from sklearn.cluster import DBSCAN, KMeans

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

#k-means
# returns the flags and the cluster of each frame
def kmeans_flags(X, k, seed):
    X = np.asarray(X, dtype=float)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    labels = km.labels_ # cluster of each frame
    dist = np.linalg.norm(X - km.cluster_centers_[labels], axis=1) # distance to its own centroid

    flags = np.zeros(len(X), dtype=bool)
    for c in range(k):
        in_c = labels == c # frames of cluster c
        d = dist[in_c]
        flags[in_c] = d > d.mean() + 3 * d.std() # threshold within the cluster

    return flags, labels

# DBSCAN on X (frames x coordinates); noise points (label = -1) are flagged
# returns the flags and the cluster of each frame
def dbscan_flags(X, eps, min_samples=4):
    X = np.asarray(X, dtype=float)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_ # -1 = noise

    return labels == -1, labels

# injection of artificial outliers (2.5)
# replaces a fraction of x by mu + s*k*sigma + q; x itself is not changed
# returns the contaminated copy and the positions replaced
def inject_outliers(x, rate, k, rng):
    x = np.asarray(x, dtype=float)
    mu, sigma = x.mean(), x.std() # before the injection

    n_out = round(rate * len(x)) # how many samples to replace
    pos = rng.choice(len(x), size=n_out, replace=False) # distinct positions
    s = rng.choice([-1, 1], size=n_out) # side, equal probability
    q = rng.uniform(0, sigma, size=n_out) # in [0, sigma)

    x_c = x.copy()
    x_c[pos] = mu + s * k * sigma + q

    return x_c, pos

# recall and precision of a detector against the known positions
# flags, truth: True/False per frame (flagged / injected)
def detection_scores(flags, truth):
    flags, truth = np.asarray(flags, bool), np.asarray(truth, bool)
    tp = (flags & truth).sum() # flagged and injected
    fp = (flags & ~truth).sum() # flagged, not injected
    fn = (~flags & truth).sum() # injected, not flagged
    return {"flagged": flags.sum(), "TP": tp, "FP": fp, "FN": fn,
            "recall": tp / (tp + fn),
            "precision": tp / (tp + fp) if tp + fp else np.nan} # nan if nothing flagged

# robust centre and spread (2.7): median and MAD (median absolute deviation)
def median_mad(r):
    r = np.asarray(r, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med)) # median of the distances to the median
    return med, mad

# modified z-score (2.7): median and MAD instead of mean and std
def modified_z(r):
    r = np.asarray(r, dtype=float)
    med, mad = median_mad(r)
    return 0.6745 * (r - med) / mad
