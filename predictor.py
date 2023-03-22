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


def main_train_multilabel(df_path, algo):
    # parameter definition

    df = pd.read_csv(df_path, usecols=['fingerprint_id', 'coord_x', 'coord_y', 'building', 'floor', 'tile'],
                     low_memory=False)
    train, test = ud.preprocess(df)
    train_data = train[['coord_x', 'coord_y']]
    train_label = train[['building', 'floor', 'tile']]
    test_label = test[['building', 'floor', 'tile']]
    test_data = test[['coord_x', 'coord_y']]

    print('Training new model...')
    params = {
        'algo_selection': algo,
        'n_neighbors': hp.choice('n_neighbors', range(1, 100)),
        'train_data': train_data,
        'train_label': train_label,
        'test_data': test_data,
        'test_label': test_label
    }

    trials = Trials()
    best = fmin(mc.clf, params, algo=tpe.suggest, max_evals=100, trials=trials)
    ud.plot_results(trials)
    model = mc.getBestModelfromTrials(trials)
    name_model = ud.save(model)
    # test model
    pred, score = mc.test(model, test[['coord_x', 'coord_y']], test_label)
    pred_df = pd.DataFrame(pred, columns=['building', 'floor', 'tile'])
    pred_df.to_excel(name_model+'.xlsx')

    return pred_df, score, name_model


# main_test predict with the model on the df and return the score
def main_test(df_path, pkl_model):
    df = pd.read_csv(df_path, usecols=['fingerprint_id', 'coord_x', 'coord_y', 'building', 'floor', 'tile'],
                     low_memory=False)
    # preprocess data with preprocessing function
    df = ud.transform(df)
    # instantiate model selected by the user
    model = ud.load(pkl_model)
    # predict
    data =df[['coord_x', 'coord_y']]
    target = df[['building', 'floor', 'tile']]
    pred = model.predict(data)
    pred_df = pd.DataFrame(pred, columns=['building', 'floor', 'tile'])
    pred_df.to_excel('test_pred.xlsx')

    score = model.score(df[['coord_x', 'coord_y']], df[['building', 'floor', 'tile']])
    final_res = metrics_eval.main(data, pred_df, target)
    # create report using create_report function
    ud.create_report(final_res)
    return pred, score, final_res


