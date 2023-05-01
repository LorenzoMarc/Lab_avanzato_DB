import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, \
    median_absolute_error, max_error


# the euclidean distance of the prediction coordinates and the target coordinates
def euclidean_distance(row):
    pred_vector = np.array([row['coord_x'], row['coord_y'], row['coord_z']])
    target_vector = np.array([row['coord_x_target'], row['coord_y_target'], row['coord_z_target']])
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
    pred_vector = np.array([row['coord_x'], row['coord_y'], row['coord_z']])
    target_vector = np.array([row['coord_x_target'], row['coord_y_target'], row['coord_z_target']])
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
    ''' based on metric selected on checkbox, apply the metric function to each row of the dataframe
    and save the result in a new column of the dataframe with the name of the metric
    The metrics are:
    1. RMSE: root mean square error of the euclidean distance between the predicted coordinates and the target coordinates
    2. 'Accuracy': calculate the accuracy of the prediction of the classes of the dataframe
    3. 'success rate': calculate the success rate of the prediction of the tile of the dataframe
        # if no metric is selected, exit the function
    '''
    if len(metrics) == 0:
        return data
    # if task is regression, calculate the metric selected
    if task == 'REGRESSION':
        if metrics['RMSE']:
            print('Calculating RMSE metric...')
            data['E'] = 0
            data['E'] = data.progress_apply(lambda row: euclidean_distance(row), axis=1)
            '''if the euclidean distance is less than 1.5 meters, the prediction is correct and the value of the column
            'class' is set to 1, otherwise it is set to 0'''
            data['class'] = np.where(data['E'] < 1.5, 1, 0)

            RMSE = np.sqrt(np.mean(data['E'] ** 2))

            # calculate the precision and recall of data['class']
            TP = data[data['class'] == 1].shape[0]
            FP = data[data['class'] == 0].shape[0]
            precision = TP / (TP + FP)
            recall = TP / data.shape[0]
            F1 = 2 * (precision * recall) / (precision + recall)
            # save the results in results_metrics
            results_metrics['metric'].append('Precision')
            results_metrics['value'].append(precision)
            results_metrics['metric'].append('Recall')
            results_metrics['value'].append(recall)
            results_metrics['metric'].append('F1')
            results_metrics['value'].append(F1)
            results_metrics['metric'].append('RMSE')
            results_metrics['value'].append(RMSE)
        if metrics['Multi']:
            print('Calculations multiple metrics...')
            # sklearn_metrics_regression is a function applied to each row of the dataframe
            # and save the result in columns accuracy
            data['Multi'] = data.progress_apply(lambda row: sklearn_metrics_regression(row), axis=1)
            new_columns = ['mean_absolute_error', 'mean_squared_error', 'r2_score', 'explained_variance_score',
                           'median_absolute_error', 'max_error']
            column = 'Multi'
            data = split_column(data, column, new_columns)
            data.drop(columns=['Multi'], inplace=True)
            # add metrics to results_metrics
            for metric in new_columns:
                results_metrics['metric'].append(metric)
                results_metrics['value'].append(data[metric].mean())

    # if task is classification, calculate the metric accuracy and success rate
    elif task.strip() == 'CLASSIFICATION':
        if metrics['Accuracy']:
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


# this function is used to calculate the metrics of the user prediction
def user_prediction(pred_df, metrics):
    results_metrics = {'metric': [], 'value': []}
    tqdm.pandas()
    # remove tab char from header of the dataframe
    pred_df.columns = pred_df.columns.str.replace('\t', '')
    pred_df = pred_df.replace('\t', '', regex=True)
    # for each metric set to 1 calculate the metric
    if metrics['RMSE']:
        print('Calculating RMSE metric...')
        # calculate the euclidean distance between the predicted coordinates and the target coordinates
        pred_df['E'] = 0
        pred_df['E'] = pred_df.progress_apply(lambda row: euclidean_distance(row), axis=1)
        # if the euclidean distance is less than 1.5 meters, the prediction is correct and the value of the column
        # 'class' is set to 1, otherwise it is set to 0
        pred_df['class'] = np.where(pred_df['E'] < 1.5, 1, 0)
        # calculate the precision and recall of data['class']
        # keep attention with division by zero
        TP = pred_df[pred_df['class'] == 1].shape[0]
        FP = pred_df[pred_df['class'] == 0].shape[0]
        precision = TP / (TP + FP)
        recall = TP / pred_df.shape[0]
        try:
            F1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            F1 = 0
        # save the results in results_metrics
        results_metrics = {'metric': ['Precision', 'Recall', 'F1'], 'value': [precision, recall, F1]}
        RMSE = np.sqrt(np.mean(pred_df['E'] ** 2))
        results_metrics['metric'].append('RMSE')
        results_metrics['value'].append(RMSE)
    if metrics['Multi']:
            print('Calculations multiple metrics...')
            # sklearn_metrics_regression is a function applied to each row of the dataframe
            # and save the result in columns accuracy
            pred_df['Multi'] = pred_df.progress_apply(lambda row: sklearn_metrics_regression(row), axis=1)
            new_columns = ['mean_absolute_error', 'mean_squared_error', 'r2_score', 'explained_variance_score',
                           'median_absolute_error', 'max_error']
            column = 'Multi'
            pred_df = split_column(pred_df, column, new_columns)
            pred_df.drop(columns=['Multi'], inplace=True)
            # add metrics to results_metrics
            for metric in new_columns:
                results_metrics['metric'].append(metric)
                results_metrics['value'].append(pred_df[metric].mean())
    if metrics['Accuracy']:
            print('Calculating accuracy metric...')
            pred_df['acc'] = pred_df.progress_apply(lambda row: accuracy_metric(row), axis=1)
            new_columns = ['accuracy_building', 'accuracy_floor', 'accuracy_tile']
            column = 'acc'
            pred_df = split_column(pred_df, column, new_columns)
            pred_df.drop(columns=['acc'], inplace=True)
            correct_building_sum = pred_df['accuracy_building'].sum()
            correct_floor_sum = pred_df['accuracy_floor'].sum()
            correct_tile_sum = pred_df['accuracy_tile'].sum()
            success_rate = 100 * (correct_tile_sum / len(pred_df))
            wrong_building_sum = len(pred_df) - correct_building_sum
            accuracy_tile = correct_floor_sum / len(pred_df)
            # print results
            print('Accuracy building: ', correct_building_sum / len(pred_df))
            print('Accuracy floor: ', correct_floor_sum / len(pred_df))
            print('Accuracy tile: ', accuracy_tile)
            print('Success rate: ', success_rate)
            print('Wrong building: ', wrong_building_sum)
            # add metrics to results_metrics
            results_metrics['metric'].extend(['accuracy_building', 'accuracy_floor', 'accuracy_tile',
                                              'success_rate', 'wrong_building'])
            results_metrics['value'].extend([correct_building_sum / len(pred_df), correct_floor_sum / len(pred_df),
                                             accuracy_tile, success_rate, wrong_building_sum])
    return pred_df, results_metrics
