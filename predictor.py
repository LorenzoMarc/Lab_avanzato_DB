'''This module is used to predict position of the user based on coordinates and RSS values of the APs
The first function is a preprocessing function that select dependent and independent variables:
    - dependent variable: coord_x, coord_y
    - independent variable: building, floor, tile
The second function is a main function that calls the model_class module to select the algorithm to use
and columns df['coord_x', 'coord_y'] as input
'''

import warnings
import metrics_eval
warnings.filterwarnings("ignore")
import model_class as mc
import utils_data as ud
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials


# used to find the columns of the dataframe that contains the APs
def find_columns(df):
    columns = []
    for col in df.columns:
        if 'AP' in col:
            columns.append(col)
    return columns


def main_train_multilabel(df_train, algo, metric_distance, n=10):
    aps = find_columns(df_train)
    df = df_train[['fingerprint_id', 'coord_x', 'coord_y', 'building', 'floor', 'tile'] + aps]
    print('Preprocessing data...')
    train, test = ud.preprocess(df, aps)
    print('Splitting data...')
    # KNN/WKK have to be trained on regression and classfication. Features input are APs.
    # classification: building, floor, tile
    # regression: coord_x, coord_y

    # hyperopt parameters
    params = {
        'algo_selection': algo,
        'metric_distance': metric_distance,
        'n_neighbors': hp.choice('n_neighbors', range(1, n)),
        'classification': ['building', 'floor', 'tile'],
        'regression': ['coord_x', 'coord_y'],
        'features': aps,
        'train': train,
        'test': test
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
    df = df[['fingerprint_id', 'coord_x', 'coord_y', 'building', 'floor', 'tile'] + aps]
    # preprocess data with preprocessing function
    df = ud.transform(df, aps)
    # instantiate model selected by the user
    model = ud.load(pkl_model)
    # predict
    data = df[aps]
    if task == 'classification'.upper():
        list_target = ['building', 'floor', 'tile']
    else:
        list_target = ['coord_x', 'coord_y']
    target = df[list_target]
    pred = model.predict(data)
    pred_df = pd.DataFrame(pred, columns=list_target)
    final_res, metrics = metrics_eval.main(data, pred_df, target, metrics, task)
    final_res = final_res.drop(aps, axis=1)
    metrics_df = pd.DataFrame.from_records(metrics, columns=['metric', 'value'])
    # save final_res and metrics_df in excel file
    ud.save_excel(final_res, metrics_df)

    # create report using create_report function
    ud.create_report(final_res)
    return pred, final_res


