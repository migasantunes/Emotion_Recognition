import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from mrmr import mrmr_classif

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


def nested_evaluation(X, y, groups, pipeline, param_grid, inner_cv, inner_groups=True):
    '''
    Outer loop: LeaveOneGroupOut, one held-out actor per fold.
    Inner loop: GridSearchCV over param_grid on the training actors only, split by inner_cv.
    The best model is refitted on all training actors and evaluated on the held-out actor.

    Parameters:
    - X, y, groups: as in evaluation()
    - pipeline: the pipeline whose parameters are tuned
    - param_grid: dict for GridSearchCV, e.g. {"svc__C": [...]}
    - inner_cv: inner splitter (GroupKFold, StratifiedKFold, etc.)
    - inner_groups: if False, the actor is not passed to the inner splitter, used for 4.7
    '''
    per_fold_results = list()
    y_true_all, y_pred_all = list(), list()

    labels = np.unique(y)

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        search = GridSearchCV(clone(pipeline), param_grid, scoring="f1_macro", cv=inner_cv, n_jobs=-1)
        search.fit(X[train_idx], y[train_idx], groups=groups[train_idx] if inner_groups else None)

        y_pred = search.predict(X[test_idx])
        y_pred_all.append(y_pred)
        y_true_all.append(y[test_idx])

        best_params = {name.split("__")[-1]: value for name, value in search.best_params_.items()}

        per_fold_results.append({"fold": len(per_fold_results) + 1,
                                 "held_out_actor": groups[test_idx][0],
                                 **best_params,
                                 "inner_best_score": search.best_score_,
                                 "accuracy": accuracy_score(y[test_idx], y_pred),
                                 "macro_f1": f1_score(y[test_idx], y_pred, average="macro", labels=labels)})

    predictions = pd.DataFrame({"y_true": np.concatenate(y_true_all),
                                "y_pred": np.concatenate(y_pred_all)})

    per_fold_results = pd.DataFrame(per_fold_results)

    return per_fold_results, predictions

class MRMRSelector(TransformerMixin, BaseEstimator):
    '''
    mrmr_classif wrapped as a scikit-learn transformer, so the selection is fitted
    inside the pipeline, on the training actors of each fold only.
    '''
    def __init__(self, K=10):
        self.K = K

    def fit(self, X, y):
        X = pd.DataFrame(np.asarray(X))  # columns named 0..n-1, so mrmr returns column indices
        self.selected_ = np.array(mrmr_classif(X=X, y=pd.Series(y), K=self.K, show_progress=False, n_jobs=1))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_]

    def get_support(self, indices=False):
        if indices:
            return self.selected_
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_] = True
        return mask
