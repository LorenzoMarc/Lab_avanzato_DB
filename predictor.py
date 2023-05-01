import time

import model_class as mc
import utils_data as ud
import pandas as pd
from hyperopt import hp
import warnings
import metrics_eval

warnings.filterwarnings("ignore")

'''This module is used to predict position of the user based on coordinates and RSS values of the APs
The first function is a preprocessing function that select dependent and independent variables:
    - dependent variable: coord_x, coord_y
    - independent variable: building, floor, tile
The second function is a main function that calls the model_class module to select the algorithm to use
and columns df['coord_x', 'coord_y'] as input
'''


# used to find the columns of the dataframe that contains the APs
def find_columns(df):
    columns = []
    for col in df.columns:
        if 'AP' in col:
            columns.append(col)
    return columns


def main_train_multilabel(df_train, algo, measure_distance, tuning, num_eval=20, n=10):
    aps = find_columns(df_train)
    df = df_train[['fingerprint_id', 'coord_x', 'coord_y', 'coord_z', 'building', 'floor', 'tile'] + aps]
    print('Preprocessing data...')
    train, test = ud.preprocess(df, aps)
    print('Splitting data...')
    # KNN/WKK have to be trained on regression and classification. Features input are APs.
    # classification: building, floor, tile
    # regression: coord_x, coord_y, coord_z
    # hyperopt parameters
    params = {
        'algo_selection': algo,
        'measure_distance': measure_distance,
        'n_neighbors': hp.choice('n_neighbors', range(1, n)),
        'classification': ['building', 'floor', 'tile'],
        'regression': ['coord_x', 'coord_y', 'coord_z'],
        'features': aps,
        'train': train,
        'test': test,
        'tuning': tuning,
        'n': n,
        'num_eval': num_eval
    }
    print('Training new model...')
    # this is the branch to train the classifier
    pred_df, score, name_model = mc.train_model(params, 'classification')

    # this is the branch to train the regressor
    pred_df_reg, score_reg, name_model_reg = mc.train_model(params, 'regression')

    return (pred_df, score, name_model), (pred_df_reg, score_reg, name_model_reg)


# main_test predict with the model on the df and return the score
def main_test(df_path, pkl_model, metrics, task):
    df = pd.read_csv(df_path, low_memory=False)
    aps = find_columns(df)
    df = df[['fingerprint_id', 'coord_x', 'coord_y', 'coord_z', 'building', 'floor', 'tile'] + aps]
    # preprocess data with preprocessing function
    df = ud.transform(df, aps)
    # instantiate model selected by the user
    model = ud.load(pkl_model)
    # predict
    data = df[aps]
    if task.upper() == 'classification'.upper():
        list_target = ['building', 'floor', 'tile']
    else:
        list_target = ['coord_x', 'coord_y', 'coord_z']
    target = df[list_target]
    try:
        pred = model.predict(data)
    except ValueError as e:
        print('Caught {} \nFeatures of the Dataset are different from the Features used in training'.format(e))
        time.sleep(90)
        pred= 0
    pred_df = pd.DataFrame(pred, columns=list_target)
    final_res, metrics_res = metrics_eval.main(data, pred_df, target, metrics, task)
    final_res = final_res.drop(aps, axis=1)
    metrics_df = pd.DataFrame.from_records(metrics_res, columns=['metric', 'value'])
    # save final_res and metrics_df in excel file
    #ud.save_excel(final_res, metrics_df)
    ud.save_csv(final_res, metrics_df)

    # create report using create_report function
    # ud.create_report(final_res)
    return pred, final_res


# main_test_pred evaluate the prediction and the dataset uploaded by the user
def main_test_pred(df_path, metrics):
    prediction = pd.read_csv(df_path, low_memory=False, usecols=lambda x: x in ['coord_x', 'coord_y', 'coord_z',
                                                                                'building', 'floor', 'tile',
                                                                                'coord_x_target', 'coord_y_target',
                                                                                'coord_z_target',
                                                                                'building_target', 'floor_target',
                                                                                'tile_target'])

    # select the column that are not in target

    final_res, metrics = metrics_eval.user_prediction(prediction, metrics)
    metrics_df = pd.DataFrame.from_records(metrics, columns=['metric', 'value'])
    # save final_res and metrics_df in excel file
    #ud.save_excel(final_res, metrics_df)
    ud.save_csv(final_res, metrics_df)
    return True
