from pathlib import Path
import pandas as pd
import numpy as np
import stats

local_path = Path(__file__).parent


path_to_repo = "C:\\Users\\abhil\\Documents\\Election Stats\\Political Spectrum\\refactor\\"

df = pd.read_csv(path_to_repo +
                 "ideo.csv")

demo = pd.read_csv(path_to_repo + "demo.csv")


def export_medians(medians, n_components):
    medians_df = pd.DataFrame.from_dict(medians, orient='index')
    component_labels = ["Component " + str(i)
                        for i in range(1, n_components + 1)]
    confidence_labels = ["Confidence Interval " +
                         str(i) for i in range(1, n_components + 1)]
    column_labels = ["Category", "Group"] + [val for pair in zip(component_labels, confidence_labels)
                                             for val in pair]
    medians_df.columns = column_labels
    # first row is explained variance

    return medians_df


def filter_data(filter_category, filter_group, data, weights):

    indices = np.where(filter_category == filter_group)[0]
    filtered = np.take(data, indices, axis=0)
    filtered_weights = np.extract(np.array(
        filter_category == filter_group), weights)
    return filtered, filtered_weights


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
        confidence_vals = np.zeros((num_cols))
        for j in range(num_cols):
            filtered_data, filtered_weights = filter_data(
                demo_labels, label, data[:, j], weights)
            median_vals[j] = stats.weighted_median(
                filtered_data, filtered_weights)
            confidence_vals[j] = 1.96 * stats.weighted_stdev(
                filtered_data, filtered_weights) / np.sqrt(len(filtered_weights))  # 95 % confidence interval
        row = [val for pair in zip(median_vals, confidence_vals)
               for val in pair]
        medians[str(demo_category) + ': ' + str(label)
                ] = [demo_category, label] + row

    return medians


def main():
    reduced = pd.DataFrame()
    demo = pd.DataFrame()

    reduced_path = local_path / "reduced.csv"

    with reduced_path.open() as f:
        reduced = pd.read_csv(f)

    demo_path = local_path / "demo.csv"
    with demo_path.open() as f:
        demo = pd.read_csv(f)

    weights = np.array(demo['weights'])

    data = np.array(reduced.values[:, 1:])

    num_components = data.shape[1]  # num of components in pca

    medians = {}

    for column in demo.columns[1:-2]:  # exclude index, age, and weights
        # add medians for each demographic variable to dictionary
        medians |= demo_medians(column, demo[column].fillna(
            "None"), data, weights)

    export_medians(medians, num_components).to_csv(local_path / "medians.csv")


if __name__ == "__main__":
    main()
