from math import sqrt
import numpy as np
import pandas as pd
import json

from pathlib import Path
local_path = Path(__file__).parent.parent

data = pd.read_csv(local_path /
                   "raw/anes_timeseries_2020_csv_20220210.csv")

data = data.drop(data.columns[[15, 17, 18, 19, 21, 22, 23, 25,
                 26, 27, 29, 30, 31, 33, 34, 35, 37, 38, 1508, 1509]], axis=1)
# get rid of some columns with bad data types

weights = pd.to_numeric(
    data['V200010b'], errors='coerce').replace(float('nan'), 0.0).rename('weights')
# preprocess weights column


def read_json(filename):
    json_dict = {}
    with open(local_path / filename, encoding='utf-8') as file:
        # do this so file will close if error
        json_dict = json.loads(file.read())
    return json_dict


ideology_vars = read_json("json/ideo_vars.json")
# contains all questions we will be handling, with variable names, question titles, questions, and answers


demographic_vars = read_json(local_path / "json/demo_vars.json")

demographic_values = read_json(local_path / "json/demo_vals.json")
# meaning of each value in a demographic variable, what indexes mean what categories
# value 0 in original data corresponds to value at index 0 here,

# IDEOLOGY QUESTIONS DATA CLEANING

ideo_data = data.filter(ideology_vars.keys())
# filters ideological questions
titles = {key: value['title'] for key, value in ideology_vars.items()}
questions = {key: value['question'] for key, value in ideology_vars.items()}
positive_answer = {key: value['answer']
                   for key, value in ideology_vars.items()}
ideo_data = ideo_data.rename(columns=positive_answer)

for col in ideo_data:
    ideo_data[col] = pd.to_numeric(
        ideo_data[col], errors='coerce')  # reformats data


def weighted_mean_and_std(vals, weights):
    mean = np.average(vals, weights=weights)
    var = np.average((vals-mean)**2, weights=weights)
    sum_weights = np.sum(weights)
    return (mean, sqrt(var * (sum_weights)/(sum_weights - 1.)))

# normalizes answers by z-scores and scales by weight


means = []
stdevs = []

for col_index, col in enumerate(ideo_data):
    not_null = np.array(
        [x >= 0 and x < 90 for x in ideo_data[col]]).astype(bool)
    # get indices of not missing data

    min_, max_ = np.amin(ideo_data[col][not_null]
                         ), np.amax(weights[not_null])
    ideo_data[col] = ideo_data[col].apply(
        lambda x: (x - min_)/(max_ - min_) * 6 + 1)
    # normalize range, most questions have 7 point scale

    mean, _ = weighted_mean_and_std(
        ideo_data[col][not_null], weights[not_null])
    means.append(mean)
    # stdev is innacurate since it doesnt count missing vals
    # so we will divide by stdev after setting missing vals to 0
    ideo_data[col] = ideo_data[col].apply(lambda x: (x - mean))

    # treat null values as 0s after normalization
    # since if someone didn't respond we can assume they are roughly neutral on the issue
    ideo_data[col] = ideo_data[col].mul(not_null.astype(int))
    # sets missing vals to 0, need to do this after subtracting mean

    _, stdev = weighted_mean_and_std(
        ideo_data[col][not_null], weights[not_null])
    ideo_data[col] = ideo_data[col].apply(lambda x: (x/stdev))
    stdevs.append(stdev)

# DEMOGRAPHIC QUESTIONS DATA CLEANING

demo_data = data.filter(demographic_vars.keys()).astype(str)
# since jsons cant have maps from ints to strings, we have maps from each code to a string
# e.g. "1": "Protestant"
# so we make all the integer values strings

demo_data.replace(demographic_values, inplace=True)

for col in demographic_values:
    demo_data[col] = demo_data[col].mask(
        pd.to_numeric(demo_data[col], errors='coerce').notna())

demo_data = demo_data.rename(columns=demographic_vars)

ideo_data = pd.concat([ideo_data, weights], axis=1)
demo_data = pd.concat([demo_data, weights], axis=1)

ideo_data.to_csv(local_path / 'data/ideo.csv')
demo_data.to_csv(local_path / 'data/demo.csv')
