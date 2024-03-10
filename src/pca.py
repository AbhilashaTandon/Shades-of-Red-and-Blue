from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import stats

local_path = Path(__file__).parent.parent


def weighted_PCA(X, num_components, weights):
    X -= np.mean(X, axis=0)
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


def apply_pca(X, eigenvector_subset):
    return np.dot(eigenvector_subset.T, X.T).T


def show_histograms(data, num_plots, weights):
    num_samples = data.shape[0]
    num_components = data.shape[1]
    # make sure we dont have more plots that components
    num_plots = min(num_plots, num_components)

    means = stats.weighted_mean(data, weights)
    # stdevs of principal components
    stdevs = stats.weighted_stdev(data, weights)

    fig, axs = plt.subplots(1, num_plots, sharey=True, tight_layout=True)

    bins_ = 40

    for x in range(num_plots):
        series = data[:, x]
        axs[x].hist(series, bins=bins_, weights=weights, alpha=.7)

        normal_dist_x = np.linspace(np.min(series), np.max(series), bins_)
        normal_dist_y = 1/(stdevs[x] * np.sqrt(2 * np.pi)) * \
            np.exp(-.5 * ((normal_dist_x - means[x]) / stdevs[x]) ** 2)

        # normalize total integral to num samples
        # makes histogram line up w normal dist
        normal_dist_y *= sum(weights) / sum(normal_dist_y)

        axs[x].axvline(x=means[x], color='red', ls='--', label=means[x])

        axs[x].plot(normal_dist_x, normal_dist_y)

    for id, ax in enumerate(axs):
        ax.set_title("PC " + str(id+1))

    plt.show()


def export_loadings(loadings, questions, n_components, explained_variance):
    loadings_df = {"Question": questions}
    for i in range(n_components):
        loadings_df['Component ' + str(i+1)] = loadings[:, i]

    loadings_df = pd.DataFrame(loadings_df)

    return loadings_df


def export_reduced(reduced, n_components, explained_variance):
    reduced_df = {}  # id for each sample

    for i in range(n_components):
        reduced_df['Component ' + str(i+1)] = reduced[:, i]

    reduced = pd.DataFrame(reduced_df)

    return reduced


def main():
    ideo = pd.DataFrame()
    demo = pd.DataFrame()

    ideo_path = local_path / "data/ideo.csv"

    with ideo_path.open() as f:
        ideo = pd.read_csv(f)

    demo_path = local_path / "data/demo.csv"
    with demo_path.open() as f:
        demo = pd.read_csv(f)

    print(ideo.head(5))

    weights = np.array(ideo['weights'])

    data = np.array(ideo.drop(columns='weights').values[:, 1:])

    num_components = 10

    x_reduced, loadings, explained_variance = weighted_PCA(
        data, num_components, weights)

    x_reduced = stats.percentile(x_reduced, weights)

    # show_histograms(x_reduced, 10, weights)

    export_reduced(x_reduced, num_components,
                   explained_variance).to_csv(local_path / "data/reduced.csv")
    export_loadings(loadings, ideo.columns[1:-1], num_components,
                    explained_variance).to_csv(local_path / "data/loadings.csv")


if (__name__ == "__main__"):
    main()
