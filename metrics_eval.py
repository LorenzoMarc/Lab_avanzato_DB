import numpy as np
import pandas as pd


# function calculate the euclidean distance between two vectors of three elements
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


# this function metric_gen is used to generate three metrics for each row of the dataframe
# the first metric 'hierch_naive' evaluate in hierarchical way the prediction, if any of the prediction is wrong,
# the score is 0 and save the score. Otherwise, the score is 1.
# The second metric 'hierarchical_with_penalty' evaluate in hierarchical way the prediction:
    # if the building prediction is correct the score is 1, else score is 0
    # if floor prediction is correct the score is 0.75, else score is 0
    # if tile prediction is correct the score is 1.25, else score is 0
# The third metric 'optimal_distance' calculate the distance from the optimal vector (1,1,1)
# and the prediction vector. The values of the prediction vector are calculated as the hierarchical_with_penalty metric.
def metric_gen(row):
    # hierarchical_naive
    hierch_naive = 0
    hierch_penalty = 0
    optimal_distance = [0, 0, 0]
    if row['building'] == row['building_target']:
        hierch_naive = 1
        hierch_penalty += 1
        optimal_distance[0] = 1
    else:
        hierch_naive = 0
        hierch_penalty += 0
        optimal_distance[0] = 0
    if row['floor'] == row['floor_target']:
        hierch_naive = 1
        hierch_penalty += 0.75
        optimal_distance[1] = 1
    else:
        hierch_naive = 0
        hierch_penalty += 0
        optimal_distance[1] = 0
    if row['tile'] == row['tile_target']:
        hierch_naive = 1
        hierch_penalty += 1.25
        optimal_distance[2] = 1
    else:
        hierch_naive = 0
        hierch_penalty += 0
        optimal_distance[2] = 0
    # optimal_distance
    optimal_vector = np.array([1, 1, 1])
    pred_vector = np.array(optimal_distance)
    pred_distance = euclidean_distance(optimal_vector, pred_vector)
    return hierch_naive, hierch_penalty, pred_distance


def main(data, pred, target):
    # rename column of target to avoid confusion
    target = target.rename(columns={'building': 'building_target', 'floor': 'floor_target', 'tile': 'tile_target'})
    # merge data, pred and target
    data = data.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    target = target.reset_index(drop=True)
    data = pd.concat([data, pred, target], axis=1)
    for row in data.iterrows():
        # apply metric_gen to each row of the dataframe
        data.loc[row[0], 'hierch_naive'], data.loc[row[0], 'hierarchical_with_penalty'], data.loc[
            row[0], 'optimal_distance'] = metric_gen(row[1])
    return data
