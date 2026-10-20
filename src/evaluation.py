from sklearn.base import clone
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

def evaluation(X, y, cv, pipeline, groups=None):
    '''
    Parameters:
    - X: numpy array of shape (n_samples, n_features) containing the feature vectors for each sample
    - y: numpy array of shape (n_samples,) containing the labels for each sample
    - cv: cross-validation generator (StratifiedShuffleSplit, LeaveOneGroupOut, etc.)
    - pipeline: The pipeline to be used for evaluation
    - groups: numpy array of shape (n_samples,) indicating the group for each sample
    '''
    per_fold_results = list()
    y_true_all, y_pred_all = list(), list()

    labels = np.unique(y)

    for train_idx, test_idx in cv.split(X, y, groups):
        pipe_train = clone(pipeline)
        pipe_train.fit(X[train_idx], y[train_idx])
        
        y_pred = pipe_train.predict(X[test_idx])
        y_pred_all.append(y_pred)
        y_true_all.append(y[test_idx])

        accuracy = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, average="macro", labels=labels)

        actor = None
        if groups is not None:
            actors = np.unique(groups[test_idx])
            if len(actors) == 1:
                actor = actors[0]

        per_fold_results.append({"fold": len(per_fold_results) + 1,
                                 "accuracy": accuracy,
                                 "macro_f1": f1,
                                 "held_out_actor": actor})

    predictions = pd.DataFrame({"y_true": np.concatenate(y_true_all),
                                "y_pred": np.concatenate(y_pred_all)})
    
    per_fold_results = pd.DataFrame(per_fold_results)

    conf_matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=labels)

    return per_fold_results, predictions, conf_matrix