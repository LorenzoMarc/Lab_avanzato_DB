'''this module has the function to fit the model selected by the user and return the predicted tile'''

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from hyperopt import STATUS_OK
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
import utils_data as ud

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                        if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_min_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_min_loss]
    return best_trial_obj['result']['Trained_Model']


def clf(parameters_list):
    params = parameters_list[0]
    task = parameters_list[1]
    algo = params['algo_selection']
    aps = [ap for ap in params['features']]
    train_features = params['train'][aps]
    test_features = params['test'][aps]
    labels_task = [lab for lab in params[task]]
    train_labels = params['train'][labels_task]
    test_labels = params['test'][labels_task]

    dict_algo = {
        'KNN': {
            'classification': MultiOutputClassifier(
                KNeighborsClassifier(n_neighbors=params['n_neighbors'], metric=params['metric_distance']),
                n_jobs=-1),
            'regression': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=params['n_neighbors'],
                                                                   metric=params['metric_distance']), n_jobs=-1)
        },
        'WKNN': {
            'classification':  MultiOutputClassifier(KNeighborsClassifier(n_neighbors=params['n_neighbors'],
                                                   metric=params['metric_distance'], weights='distance'),  n_jobs=-1),
            'regression': MultiOutputRegressor(KNeighborsRegressor(n_neighbors=params['n_neighbors'],
                                              metric=params['metric_distance'], weights='distance'), n_jobs=-1)
        }
    }

    model = dict_algo[algo][task]
    model.fit(train_features, train_labels)
    pred = model.predict(test_features)
    score = model.score(test_features, test_labels)
    print('Score for {} with multilabel {} is: {}'.format(algo,task, score))
    return {'loss': -score, 'status': STATUS_OK, 'predicitons': pred, 'Trained_Model': model}


# test the model on the test data and return the score
def test(model, test_data, test_label):
    # Make a prediction using the optimized model
    prediction = model.predict(test_data)
    # Report the accuracy of the classifier on a given set of data
    score = model.score(test_data, test_label)
    return prediction, score


# given the task it select the right model and train it
def train_model(params, task):
    trials = Trials()
    # get the model
    model = fmin(clf, [params, task], algo=tpe.suggest, max_evals=10, trials=trials)
    best_model = getBestModelfromTrials(trials)
    # get the prediction and the score
    labels_task = [lab for lab in params[task]]
    aps = [ap for ap in params['features']]
    prediction, score = test(best_model, params['test'][aps], params['test'][labels_task])
    # save model
    ud.save(best_model, task)
    ud.plot_results(trials, task, params['algo_selection'])

    return prediction, score, best_model
