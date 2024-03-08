from pathlib import Path
import pandas as pd
import numpy as np
import stats

local_path = Path(__file__).parent.parent

df = pd.read_csv(local_path /
                 "data/ideo.csv")

demo = pd.read_csv(local_path / "data/demo.csv")


def export_means(means, n_components):
    means_df = pd.DataFrame.from_dict(means, orient='index')
    component_labels = ["Component " + str(i)
                        for i in range(1, n_components + 1)]
    confidence_labels = ["Confidence Interval " +
                         str(i) for i in range(1, n_components + 1)]
    var_labels = ["Variance of Comp. " + str(i)
                  for i in range(1, n_components + 1)]
    var_of_var_labels = ["Variance of Sample Variance of Comp." + str(i)
                         for i in range(1, n_components + 1)]
    column_labels = ["Category", "Group"] + [val for pair in zip(component_labels, confidence_labels, var_labels, var_of_var_labels)
                                             for val in pair]
    means_df.columns = column_labels
    # first row is explained variance

    return means_df


def filter_data(filter_category, filter_group, data, weights):

    indices = np.where(filter_category == filter_group)[0]
    filtered = np.take(data, indices, axis=0)
    filtered_weights = np.extract(np.array(
        filter_category == filter_group), weights)
    return filtered, filtered_weights


def demo_means(demo_category, demo_labels, data, weights):
    all_labels = sorted(list(set(demo_labels)))
    num_cols = data.shape[1]
    assert (num_cols != len(demo_labels))
    # this is a weird mess of a line

    # demo.shape should be (num samples, 1)
    # data.shape should be (num samples, num_cols)

    means = {}

    for label in all_labels:
        mean_vals = np.zeros((num_cols))
        confidence_vals = np.zeros((num_cols))
        sample_vars = np.zeros((num_cols))
        var_confidence_values = np.zeros((num_cols))

        n = len(weights)

        for j in range(num_cols):
            filtered_data, filtered_weights = filter_data(
                demo_labels, label, data[:, j], weights)
            mean_vals[j] = stats.weighted_mean(
                filtered_data, filtered_weights)
            confidence_vals[j] = 1.96 * stats.weighted_stdev(
                filtered_data, filtered_weights) / np.sqrt(n)  # 95 % confidence interval
            sample_vars[j] = stats.nth_central_moment(
                filtered_data, filtered_weights, 2) * (n / (n - 1))  # sample var is E(x - mu)^2 / (n-1)
            var_confidence_values[j] = 1.96 * stats.nth_central_moment(
                filtered_data, filtered_weights, 4) - sample_vars[j] * (n-3) / (n * (n-1))
        row = [val for pair in zip(mean_vals, confidence_vals, sample_vars, var_confidence_values)
               for val in pair]
        means[str(demo_category) + ': ' + str(label)
              ] = [demo_category, label] + row

    return means


def main():
    reduced = pd.DataFrame()
    demo = pd.DataFrame()

    reduced_path = local_path / "data/reduced.csv"

    with reduced_path.open() as f:
        reduced = pd.read_csv(f)

    demo_path = local_path / "data/demo.csv"
    with demo_path.open() as f:
        demo = pd.read_csv(f)

    weights = np.array(demo['weights'])

    data = np.array(reduced.values[:, 1:])

    num_components = data.shape[1]  # num of components in pca

    print(demo.columns)

    means = {}

    for column in demo.columns[1:-2]:  # exclude index, age, and weights
        # add means for each demographic variable to dictionary
        means |= demo_means(column, demo[column].fillna(
            "None"), data, weights)

    export_means(means, num_components).to_csv(local_path / "data/means.csv")


if __name__ == "__main__":
    main()
