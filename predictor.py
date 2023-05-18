import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

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
def find_aps(df):
    columns = []
    for col in df.columns:
        if 'AP' in col:
            columns.append(col)
    return columns


def main_train_multilabel(df_train, algo, measure_distance, tuning, num_eval=20, n=10, test_size=0.2):

    # find the columns of the dataframe that contains the APs
    aps = find_aps(df_train)
    df = df_train[['fingerprint_id', 'coord_x', 'coord_y', 'coord_z', 'building', 'floor', 'tile'] + aps]
    print('Preprocessing data...')
    enc = OrdinalEncoder()
    # fill numeric missing APs with 100
    df[aps] = df[aps].fillna(100)
    df[['coord_x', 'coord_y', 'coord_z']] = df[['coord_x', 'coord_y', 'coord_z']].fillna(0)
    # fill categorical missing values with 'missing'
    df[['building', 'floor', 'tile']] = df[['building', 'floor', 'tile']].fillna('missing')
    df[['building','floor', 'tile']] = enc.fit_transform(df[['building', 'floor', 'tile']])
    if test_size > 0.99:
        print('error the test size cannot be more then .99')
    train, test = train_test_split(df, test_size=test_size, random_state=42)
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

    pred_clf, score, name_model = mc.train_model(params, 'classification')
    pred_reg, score_reg, name_model_reg = mc.train_model(params, 'regression')
    decoded_clf = enc.inverse_transform(pred_clf)

    return (decoded_clf, score, name_model), (pred_reg, score_reg, name_model_reg)


# main_test predict with the model on the df and return the score
def main_test(df_path, pkl_model, metrics, task):
    df = pd.read_csv(df_path, low_memory=False)
    aps = find_aps(df)
    df = df[['fingerprint_id', 'coord_x', 'coord_y', 'coord_z', 'building', 'floor', 'tile'] + aps]
    df[aps] = df[aps].fillna(100)

    df[['coord_x', 'coord_y', 'coord_z']] = df[['coord_x', 'coord_y', 'coord_z']].fillna(0)
    # fill categorical missing values with 'missing'
    df[['building', 'floor', 'tile']] = df[['building', 'floor', 'tile']].fillna('missing')
    enc = OrdinalEncoder()
    df[['building','floor', 'tile']] = enc.fit_transform(df[['building', 'floor', 'tile']])

    if task.upper() == 'classification'.upper():
        list_target = ['building', 'floor', 'tile']
        non_target = ['coord_x', 'coord_y', 'coord_z']
        not_target = df[non_target]
    else:
        list_target = ['coord_x', 'coord_y', 'coord_z']
        non_target = ['building', 'floor', 'tile']
        not_target = df[non_target]

    target = df[list_target]
    model = ud.load(pkl_model)
    features = ud.features_check(df, model.aps)
    common_aps = features.columns
    try:
        pred = model.predict(features)
    except ValueError as e:
        print('Caught {} \nFeatures of the Dataset are different from the Features used in training'.format(e))
        time.sleep(90)
        pred = 0
    pred_df = pd.DataFrame(pred, columns=list_target)
    # evaluate the prediction made by the model with the metrics selected
    final_res, metrics_res = metrics_eval.main(features, pred_df, target, not_target,
                                               metrics,
                                               task)
    final_res = final_res.drop(common_aps, axis=1)
    metrics_df = pd.DataFrame.from_records(metrics_res, columns=['metric', 'value'])
    final_res[['building', 'floor', 'tile']] = enc.inverse_transform(final_res[['building', 'floor', 'tile']])
    final_res[['building_target', 'floor_target', 'tile_target']] =\
        enc.inverse_transform(final_res[['building_target', 'floor_target', 'tile_target']])
    # save final_res and metrics_df in excel file
    ud.save_excel(final_res, metrics_df)
    #ud.save_csv(final_res, metrics_df)

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
