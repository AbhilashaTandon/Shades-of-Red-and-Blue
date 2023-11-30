from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

path_to_repo = "C:\\Users\\abhil\\Documents\\Election Stats\\Political Spectrum\\refactor\\"

df = pd.read_csv(path_to_repo +
                 "ideo.csv")

weights = np.array(df['weights'])

# get rid of weights, [:, 1:] to get rid of ids
data = np.array(df.drop(columns='weights').values[:, 1:])


def weighted_PCA(X, num_components, weights):
    covar = X.T @ np.diag(weights) @ X  # weighted covariance matrix

    eigenvals, eigenvecs = np.linalg.eigh(covar)

    sorted_idxs = np.argsort(eigenvals)[::-1]  # sorted descending
    sorted_eigenvals = eigenvals[sorted_idxs]
    sorted_eigenvecs = eigenvecs[:, sorted_idxs]

    explained_variance = sorted_eigenvals / np.sum(sorted_eigenvals)
    explained_variance = explained_variance[:num_components]

    # only need the first n components
    eigenvector_subset = sorted_eigenvecs[:, :num_components]

    X_reduced = np.dot(eigenvector_subset.T,
                       X.T).T

    return X_reduced, eigenvector_subset, explained_variance


num_components = 10
x_reduced, loadings, explained_variance = weighted_PCA(
    data, num_components, weights)

question_dataframe = {"Question": df.columns[1:-1]}


for i in range(num_components):
    question_dataframe['Component ' + str(i+1)] = loadings[:, i]

questions = pd.DataFrame(question_dataframe)

questions.to_csv(path_to_repo + 'questions.csv')
