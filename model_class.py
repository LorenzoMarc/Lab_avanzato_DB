'''this module has the function to fit the model selected by the user and return the predicted tile'''
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import STATUS_OK
import numpy as np


def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['Trained_Model']


def clf(params):
    algo = params['algo_selection']
    dict_algo = {'KNN': KNeighborsClassifier(n_neighbors=params['n_neighbors']),
                 'svc': SVC(kernel="linear", C=0.025),
                 'dt': DecisionTreeClassifier(max_depth=5),
                 'rf': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                 'ada': AdaBoostClassifier(), 'mlp': MLPClassifier(alpha=1, max_iter=1000)}
    model_sel = dict_algo[algo]
    model = MultiOutputClassifier(model_sel, n_jobs=-1)
    model.fit(params['train_data'], params['train_label'])
    pred = model.predict(params['test_data'])
    score = model.score(params['test_data'], params['test_label'])
    print('Score for {} with multilabel classification is: {}'.format(algo, score))
    return {'loss': -score, 'status': STATUS_OK, 'predicitons': pred, 'Trained_Model': model}


# test the model on the test data and return the score
def test(model, test_data, test_label):
    # Make a prediction using the optimized model
    prediction = model.predict(test_data)
    # Report the accuracy of the classifier on a given set of data
    score = model.score(test_data, test_label)
    return prediction, score
