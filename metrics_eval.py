import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, \
    median_absolute_error, max_error


# the euclidean distance of the prediction coordinates and the target coordinates
def euclidean_distance(row):
    pred_vector = np.array([row['coord_x'], row['coord_y'], row['coord_z']])
    try:
        target_vector = np.array([row['coord_x_target'], row['coord_y_target'], row['coord_z_target']])
    except:
        target_vector = np.array([0, 0, 0])
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


def two_d_error(row):
    # check if row['coord_z'] is equal to coord_z_target
    if row['coord_z'] == row['coord_z_target']:

        pred_vector = np.array([row['coord_x'], row['coord_y']])
        try:
            target_vector = np.array([row['coord_x_target'], row['coord_y_target']])
        except:
            target_vector = np.array([0, 0])
        return np.sqrt(np.sum((pred_vector - target_vector) ** 2))
    else:
        return 0


# this function is used to calculate the accuracy of the prediction of the floor
def fdp_metric(row):
    if row['coord_z'] == row['coord_z_target']:
        return 1
    else:
        return 0


def main(features, pred, target, not_target, metrics, task):
    tqdm.pandas()
    results_metrics = {'metric': [], 'value': []}
    # for each column of target add '_target' to the name
    target.columns = [str(col) + '_target' for col in target.columns]
    features = features.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    target = target.reset_index(drop=True)
    original = pd.concat([target, not_target, features], axis=1)
    data = pd.concat([pred, original], axis=1)

    ''' based on metric selected on checkbox, apply the euclidean distance function to each row of the dataframe
    and save the result in a new column of the dataframe
     if no metric is selected, exit the function
    '''
    if len(metrics) == 0:
        return data
    # if task is regression, calculate the metric selected in regression tab
    if task == 'REGRESSION':
        # calculate the euclidean distance and the RMSE
        data['E'] = 0
        data['E'] = data.progress_apply(lambda row: euclidean_distance(row), axis=1)
        '''if the euclidean distance is less than 1.5 meters, the prediction is correct and the value of the column
        'class' is set to 1, otherwise it is set to 0'''
        data['class'] = np.where(data['E'] < 1.5, 1, 0)
        if metrics['RMSE']:
            print('Calculating RMSE metric...')
            RMSE = np.sqrt(np.mean(data['E'] ** 2))
            results_metrics['metric'].append('RMSE')
            results_metrics['value'].append(RMSE)
        # calculate the precision, recall and F1 score
        if metrics['Precision']:
            print('Calculating Precision metric...')
            TP = data[data['class'] == 1].shape[0]
            FP = data[data['class'] == 0].shape[0]
            precision = TP / (TP + FP)
            results_metrics['metric'].append('Precision')
            results_metrics['value'].append(precision)
        if metrics['Recall']:
            print('Calculating Recall metric...')
            TP = data[data['class'] == 1].shape[0]
            recall = TP / data.shape[0]
            results_metrics['metric'].append('Recall')
            results_metrics['value'].append(recall)
        if metrics['F1']:
            print('Calculating F1 metric...')
            TP = data[data['class'] == 1].shape[0]
            FP = data[data['class'] == 0].shape[0]
            precision = TP / (TP + FP)
            recall = TP / data.shape[0]
            F1 = 2 * (precision * recall) / (precision + recall)
            results_metrics['metric'].append('F1')
            results_metrics['value'].append(F1)
        if metrics['MAE']:
            print('Calculations MAE metrics...')
            # calculate the MAE of data['E'] and save the result in results_metrics
            MAE = data['E'].mean()
            results_metrics['metric'].append('MAE')
            results_metrics['value'].append(MAE)
        if metrics['MSE']:
            MSE = mean_squared_error(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('MSE')
            results_metrics['value'].append(MSE)
        if metrics['R2']:
            R2 = r2_score(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('R2')
            results_metrics['value'].append(R2)
        if metrics['EVS']:
            EVS = explained_variance_score(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('EVS')
            results_metrics['value'].append(EVS)
        if metrics['MedAE']:
            MedAE = median_absolute_error(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('MedAE')
            results_metrics['value'].append(MedAE)
        if metrics['ME']:
            ME = max_error(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('ME')
            results_metrics['value'].append(ME)
        # if metrics is 2d error calculate:
        # coord_z is the coordinate of the floor, if predicted z is equal to target z, the prediction is correct
        # then calculate the distance between the predicted and the target coord_x and coord_y,
        # if the z is not correct, the distance is not calculated
        if metrics['2D Error']:
            print('Calculating 2D Error metric...')
            data['2D Error'] = data.progress_apply(lambda row: two_d_error(row), axis=1)
            # calculate the mean of the 2D Error if and only if the z is correct
            mean_2d_error = data[data['2D Error'] != 0]['2D Error'].mean()
            results_metrics['metric'].append('2D Error')
            results_metrics['value'].append(mean_2d_error)
        # if 'fdp' is in metrics, calculate the percentage of correct predictions of the floor
        if metrics['fdp']:
            print('Calculating fdp metric...')
            # calculate data['fdp'] and save the result in results_metrics
            data['fdp'] = data.progress_apply(lambda row: fdp_metric(row), axis=1)
            correct_floor_sum = data['fdp'].sum()
            fdp = 100 * (correct_floor_sum / len(data))
            results_metrics['metric'].append('fdp')
            results_metrics['value'].append(fdp)
        # if SD is in metrics, calculate the standard deviation of the euclidean distance
        if metrics['SD']:
            print('Calculating SD metric...')
            SD = data['E'].std()
            results_metrics['metric'].append('SD')
            results_metrics['value'].append(SD)

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
    data = pred_df.copy()
    if not metrics['Accuracy']:
        data['E'] = 0
        data['E'] = data.progress_apply(lambda row: euclidean_distance(row), axis=1)
        '''if the euclidean distance is less than 1.5 meters, the prediction is correct and the value of the column
        'class' is set to 1, otherwise it is set to 0'''
        data['class'] = np.where(data['E'] < 1.5, 1, 0)
        if metrics['RMSE']:
            print('Calculating RMSE metric...')
            RMSE = np.sqrt(np.mean(data['E'] ** 2))
            results_metrics['metric'].append('RMSE')
            results_metrics['value'].append(RMSE)
        # calculate the precision, recall and F1 score
        if metrics['Precision']:
            print('Calculating Precision metric...')
            TP = data[data['class'] == 1].shape[0]
            FP = data[data['class'] == 0].shape[0]
            precision = TP / (TP + FP)
            results_metrics['metric'].append('Precision')
            results_metrics['value'].append(precision)
        if metrics['Recall']:
            print('Calculating Recall metric...')
            TP = data[data['class'] == 1].shape[0]
            recall = TP / data.shape[0]
            results_metrics['metric'].append('Recall')
            results_metrics['value'].append(recall)
        if metrics['F1']:
            print('Calculating F1 metric...')
            TP = data[data['class'] == 1].shape[0]
            FP = data[data['class'] == 0].shape[0]
            precision = TP / (TP + FP)
            recall = TP / data.shape[0]
            F1 = 2 * (precision * recall) / (precision + recall)
            results_metrics['metric'].append('F1')
            results_metrics['value'].append(F1)
        if metrics['MAE']:
            print('Calculations MAE metric...')
            # calculate the MAE of data['E'] and save the result in results_metrics
            MAE = data['E'].mean()
            results_metrics['metric'].append('MAE')
            results_metrics['value'].append(MAE)
        if metrics['MSE']:
            MSE = mean_squared_error(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('MSE')
            results_metrics['value'].append(MSE)
        if metrics['R2']:
            R2 = r2_score(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('R2')
            results_metrics['value'].append(R2)
        if metrics['EVS']:
            EVS = explained_variance_score(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('EVS')
            results_metrics['value'].append(EVS)
        if metrics['MedAE']:
            MedAE = median_absolute_error(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('MedAE')
            results_metrics['value'].append(MedAE)
        if metrics['ME']:
            ME = max_error(data['E'], np.zeros(data.shape[0]))
            results_metrics['metric'].append('ME')
            results_metrics['value'].append(ME)
        # if metrics is 2d error calculate:
        # coord_z is the coordinate of the floor, if predicted z is equal to target z, the prediction is correct
        # then calculate the distance between the predicted and the target coord_x and coord_y,
        # if the z is not correct, the distance is not calculated
        if metrics['2D Error']:
            print('Calculating 2D Error metric...')
            data['2D Error'] = data.progress_apply(lambda row: two_d_error(row), axis=1)
            # calculate the mean of the 2D Error if and only if the z is correct
            mean_2d_error = data[data['2D Error'] != 0]['2D Error'].mean()
            results_metrics['metric'].append('2D Error')
            results_metrics['value'].append(mean_2d_error)
        # if 'fdp' is in metrics, calculate the percentage of correct predictions of the floor
        if metrics['fdp']:
            print('Calculating fdp metric...')
            # calculate data['fdp'] and save the result in results_metrics
            data['fdp'] = data.progress_apply(lambda row: fdp_metric(row), axis=1)
            correct_floor_sum = data['fdp'].sum()
            fdp = 100 * (correct_floor_sum / len(data))
            results_metrics['metric'].append('fdp')
            results_metrics['value'].append(fdp)

    if metrics['Accuracy']:
        print('Calculating accuracy metric...')
        # ordinal encoding of classification labels (building, floor, tile) and target labels (building_target, floor_target, tile_target)
        pred_df['building'] = pred_df['building'].astype('category')
        pred_df['floor'] = pred_df['floor'].astype('category')
        pred_df['tile'] = pred_df['tile'].astype('category')
        pred_df['building_target'] = pred_df['building_target'].astype('category')
        pred_df['floor_target'] = pred_df['floor_target'].astype('category')
        pred_df['tile_target'] = pred_df['tile_target'].astype('category')

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
