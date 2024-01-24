from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


path_to_repo = "C:\\Users\\abhil\\Documents\\Election Stats\\Political Spectrum\\refactor\\"

df = pd.read_csv(path_to_repo +
                 "ideo.csv")

demo = pd.read_csv(path_to_repo + "demo.csv")


def export_questions_data(loadings, n_components, file_name, explained_variance):
    question_dataframe = {"Question": df.columns[1:-1]}
    for i in range(n_components):
        question_dataframe['Component ' + str(i+1)] = loadings[:, i]

    questions = pd.DataFrame(question_dataframe)

    questions.loc[0] = ['Explained Variance'] + \
        list(explained_variance)  # first row is explained variance

    questions.to_csv(path_to_repo + file_name + '.csv')


def export_medians(medians, n_components, file_name, explained_variance):
    df = pd.DataFrame.from_dict(medians, orient='index')
    df.columns = ["Component " +
                  str(i) for i in range(1, n_components + 1)]
    # first row is explained variance
    df.loc['Explained Variance'] = list(explained_variance)

    df.to_csv(path_to_repo + file_name + '.csv')


def filter_data(filter_category, filter_group, data, weights):

    indices = np.where(filter_category == filter_group)[0]
    filtered = np.take(data, indices, axis=0)
    filtered_weights = np.extract(np.array(
        filter_category == filter_group), weights)
    return filtered, filtered_weights


def weighted_median(values, weights):
    cumsums = np.cumsum(weights)
    halfway = np.sum(weights)/2
    median_idx = (np.abs(cumsums - halfway)).argmin()
    return sorted(values)[median_idx]


def weighted_mean(values, weights):
    return np.average(values, weights=weights, axis=0)


def weighted_stdev(values, weights):
    means = weighted_mean(values, weights)
    return np.sqrt(np.average((values-means)**2, weights=weights, axis=0))


def demo_medians(demo_category, demo_labels, data, weights):
    all_labels = sorted(list(set(demo_labels)))
    num_cols = data.shape[1]
    assert (num_cols != len(demo_labels))
    # this is a weird mess of a line

    # demo.shape should be (num samples, 1)
    # data.shape should be (num samples, num_cols)

    medians = {}

    for label in all_labels:
        median_vals = np.zeros((num_cols))
        for j in range(num_cols):
            filtered_data, filtered_weights = filter_data(
                demo_labels, label, data[:, j], weights)
            median_vals[j] = weighted_median(filtered_data, filtered_weights)
        medians[str(demo_category) + ': ' + str(label)] = list(median_vals)

    return medians


def show_histograms(data, num_plots, weights):
    num_samples = data.shape[0]
    num_components = data.shape[1]
    num_plots = min(num_plots, num_components)

    means = weighted_mean(data, weights)
    stdevs = weighted_stdev(data, weights)  # stdevs of principal components

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

    bins_ = int(np.sqrt(num_samples))

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


def main():
    weights = np.array(df['weights'])

    # get rid of weights, [:, 1:] to get rid of ids
    data = np.array(df.drop(columns='weights').values[:, 1:])

    num_components = 10

    x_reduced, loadings, explained_variance = weighted_PCA(
        data, num_components, weights)

    means = weighted_mean(x_reduced, weights=weights)
    stdevs = weighted_stdev(x_reduced, weights=weights)

    x_reduced = ((x_reduced - means)) / stdevs

    num_samples = x_reduced.shape[0]

    export_questions_data(loadings, num_components,
                          "loadings", explained_variance)

    filtered, filtered_weights = filter_data(
        demo['PRE-POST: SUMMARY: 2020 PRESIDENTIAL VOTE'], "Joe Biden", x_reduced, weights)

    print(filtered.shape)

    show_histograms(filtered, 3, filtered_weights)

    medians = {}

    for column in demo.columns[1:-2]:  # exclude index, age, and weights
        # add medians for each demographic variable to dictionary
        medians |= demo_medians(column, demo[column].fillna(
            "None"), x_reduced, weights)

    export_medians(medians, num_components, "medians", explained_variance)

    # filter_category = demo['PRE-POST: SUMMARY: 2020 PRESIDENTIAL VOTE']

    # filtered_data, filtered_weights = filter(
    #     filter_category, 'Joe Biden', num_questions, data, weights)


if (__name__ == "__main__"):
    main()

# # PLOTTING

# demo = pd.read_csv(path_to_repo + "demo.csv")

# demo_category = demo['PRE: 7PT SCALE LIBERAL-CONSERVATIVE SELFPLACEMENT'].fillna(
#     "None")

# # get sorted list of groups in descending order
# groups = list(set(demo_category))

# occurences = {group: 0 for group in groups}
# for group in demo_category:
#     occurences[group] += 1

# groups = list(
#     dict(sorted(occurences.items(), key=lambda x: x[1], reverse=True)).keys())  # sorts groups by occurences

# colors = ['#000000', '#c95200', '#1e8a21', '#0486cd', '#d1c102',
#           '#fe8ca6', '#90133c', '#3cff92', '#283784', '#21d1e1', '#79727e']
# # most distinct colors for more frequent groups


def make_plot(embedding, num_dims, groups, colors, demo_category):

    fig, ax = plt.subplots()

    for group, color in zip(groups, colors):
        elements = np.extract(np.array(np.repeat(
            demo_category == group, num_dims)).reshape(-1, num_dims), embedding)

        ax.scatter(elements[:, 0], elements[:, 1], c=color,
                   s=40, alpha=.5, label=group)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
               fancybox=True, shadow=True, ncol=3, borderpad=.5, labelspacing=.5)
    plt.show()
