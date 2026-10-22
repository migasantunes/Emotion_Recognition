import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


def loadings_table(pca, feature_names, n_components=2):
    # one row per feature, one column per principal component (PC1, PC2, ...)
    columns = [f"PC{i + 1}" for i in range(n_components)]
    loadings = pca.components_[:n_components].T
    return pd.DataFrame(loadings, index=feature_names, columns=columns)


def measure_of(feature_name):
    # "M1_mean" -> "M1", "own_M7_vel" -> "own_M7"
    return feature_name.rsplit("_", 1)[0]


def loadings_by_measure(pca, feature_names, n_components=2):
    # share of each component that comes from each measure:
    # the squared loadings of one component add up to 1, so we sum them per measure
    table = loadings_table(pca, feature_names, n_components)
    squared = table ** 2
    measures = [measure_of(name) for name in table.index]
    return squared.groupby(measures, sort=False).sum()


def pca_components_per_fold(X, y, groups, variance):
    # number of components PCA keeps in each LOSO fold,
    # with the scaler and the PCA fitted on the training actors only
    rows = list()
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        scaler = StandardScaler()
        Z_train = scaler.fit_transform(X[train_idx])

        pca = PCA(n_components=variance)
        pca.fit(Z_train)

        rows.append({"held_out_actor": groups[test_idx][0],
                     "n_components": pca.n_components_})
    return pd.DataFrame(rows)
