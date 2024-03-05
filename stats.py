import numpy as np


def weighted_median(values, weights):
    cumsums = np.cumsum(weights)
    halfway = np.sum(weights)/2
    median_idx = (np.abs(cumsums - halfway)).argmin()
    return sorted(values)[median_idx]


def weighted_mean(values, weights):
    if (values.ndim > 1):
        return np.average(values, weights=weights, axis=0)
    return np.average(values, weights=weights, axis=0)


def weighted_stdev(values, weights):
    means = weighted_mean(values, weights)
    if (values.ndim > 1):
        return np.sqrt(np.average((values-means)**2, weights=weights, axis=0))
    return np.sqrt(np.average((values-means)**2, weights=weights))


def weighted_z_score(values, weights):
    # normalized data by mean and stdev, using weights
    means = weighted_mean(values, weights)
    stdevs = weighted_stdev(values, weights)
    return (values - means)/stdevs


def nth_central_moment(values, weights, n):
    mu = weighted_mean(values, weights=weights)
    deviations = values - mu
    nth_moment = np.power(deviations, n)
    return np.average(nth_moment, weights=weights)


def percentile(values, weights):
    out = []
    for series in values.T:
        ix = np.argsort(series)  # indices to sort weights by values

        sorted_weights = weights[ix]
        cdf = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / \
            np.sum(sorted_weights)  # like a cdf

        reverse_idx = np.argsort(ix)
        out.append(cdf[reverse_idx])
    return np.array(out).T
