import numpy as np
import pandas as pd
from tqdm import tqdm


# the euclidean distance of the prediction coordinates and the target coordinates
def euclidean_distance(row):
    pred_vector = np.array([row['coord_x'], row['coord_y']])
    target_vector = np.array([row['coord_x_target'], row['coord_y_target']])
    return np.sqrt(np.sum((pred_vector - target_vector) ** 2))


# accuracy_metric calculate the accuracy of the prediction of the building of the dataframe
# by looking at the number of correct predictions against the wrong ones
def accuracy_metric(row):
    if row['building'] == row['building_target'] \
            and row['floor'] == row['floor_target'] \
            and row['tile'] == row['tile_target']:
        return 1, 1, 1
    elif row['building'] == row['building_target'] and row['floor'] == row['floor_target']:
        return 1, 1, 0
    elif row['building'] == row['building_target']:
        return 1, 0, 0
    else:
        return 0, 0, 0


# this function split_column is used to split the column
# 'acc' into 3 columns 'accuracy_building',
# 'accuracy_floor' and 'accuracy_tile'
def split_column(df, column, new_columns):
    df[new_columns] = pd.DataFrame(df[column].values.tolist(), index=df.index)
    return df


# this function use the sklear metrics of regression task to calculate 6 sklearn metrics
# 1. mean absolute error
# 2. mean squared error
# 3. r2 score
# 4. explained variance score
# 5. median absolute error
# 6. max error
def sklearn_metrics_regression(row):
    pred_vector = np.array([row['coord_x'], row['coord_y']])
    target_vector = np.array([row['coord_x_target'], row['coord_y_target']])
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, \
        median_absolute_error, max_error
    return mean_absolute_error(target_vector, pred_vector), \
           mean_squared_error(target_vector, pred_vector), \
           r2_score(target_vector, pred_vector), \
           explained_variance_score(target_vector, pred_vector), \
           median_absolute_error(target_vector, pred_vector), \
           max_error(target_vector, pred_vector)


def main(data, pred, target, metrics, task):
    tqdm.pandas()
    results_metrics = {'metric': [], 'value': []}
    # for each column of target add '_target' to the name
    target.columns = [str(col) + '_target' for col in target.columns]
    # merge data, pred and target
    data = data.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    target = target.reset_index(drop=True)
    data = pd.concat([data, pred, target], axis=1)
    '''TODO: based on metric selected on checkbox, apply the metric function to each row of the dataframe
    and save the result in a new column of the dataframe with the name of the metric
    The metrics are:
    1. RMSE: root mean square error of the euclidean distance between the predicted coordinates and the target coordinates
    2. 'accuracy': calculate the accuracy of the prediction of the building of the dataframe
    3. 'success rate': calculate the success rate of the prediction of the tile of the dataframe
        # if no metric is selected, exit the function
    '''
    if len(metrics) == 0:
        return data
    # if task is regression, calculate the metric RMSE
    if task == 'REGRESSION':
        # if 'RMSE' in metrics calculate the RMSE metric of the euclidean distance between the predicted coordinates
        # and the target coordinates
        if 'RMSE' in metrics:
            print('Calculating RMSE metric...')
            data['E'] = 0
            data['E'] = data.progress_apply(lambda row: euclidean_distance(row), axis=1)
        RMSE = np.sqrt(np.mean(data['E'] ** 2))
        results_metrics['metric'].append('RMSE')
        results_metrics['value'].append(RMSE)
        if 'Multi' in metrics:
            print('Calculations multiple metrics...')
            # sklearn_metrics_regression is a function applied to each row of the dataframe
            # and save the result in columns accuracy
            data['Multi'] = data.progress_apply(lambda row: sklearn_metrics_regression(row), axis=1)
            new_columns = ['mean_absolute_error', 'mean_squared_error', 'r2_score', 'explained_variance_score', 'median_absolute_error', 'max_error']
            column = 'Multi'
            data = split_column(data, column, new_columns)
            data.drop(columns=['Multi'], inplace=True)
            # add metrics to results_metrics
            for metric in new_columns:
                results_metrics['metric'].append(metric)
                results_metrics['value'].append(data[metric].mean())

    # if task is classification, calculate the metric accuracy and success rate
    elif task.strip() == 'CLASSIFICATION':
        # note: accuracy TP+TN/TP+TN+FP+FN (building) but suffer from class imbalance
        if 'Accuracy' in metrics:
            print('Calculating accuracy metric...')

            data['acc'] = data.progress_apply(lambda row: accuracy_metric(row), axis=1)
            new_columns = ['accuracy_building', 'accuracy_floor', 'accuracy_tile']
            column = 'acc'
            data = split_column(data, column, new_columns)
            data.drop(columns=['acc'], inplace=True)
            correct_building_sum = data['accuracy_building'].sum()
            correct_floor_sum = data['accuracy_floor'].sum()
            correct_tile_sum = data['accuracy_tile'].sum()

            success_rate = 100 * (correct_tile_sum / len(data))
            wrong_building_sum = len(data) - correct_building_sum
            accuracy_tile = correct_floor_sum / len(data)
            # print results
            print('Accuracy building: ', correct_building_sum / len(data))
            print('Accuracy floor: ', correct_floor_sum / len(data))
            print('Accuracy tile: ', accuracy_tile)
            print('Success rate: ', success_rate)
            print('Wrong building: ', wrong_building_sum)
            # add metrics to results_metrics
            results_metrics['metric'].extend(['accuracy_building', 'accuracy_floor', 'accuracy_tile',
                                              'success_rate', 'wrong_building'])
            results_metrics['value'].extend([correct_building_sum / len(data), correct_floor_sum / len(data),
                                             accuracy_tile, success_rate, wrong_building_sum])

    return data, results_metrics
