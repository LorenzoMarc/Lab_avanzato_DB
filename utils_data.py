import pickle as pkl
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split  # for splitting the data into train and test samples
from sklearn.preprocessing import MinMaxScaler  # for feature scaling
from sklearn.preprocessing import OrdinalEncoder  # to encode categorical variables
from matplotlib.pyplot import savefig, subplots
import pandas as pd


# from pandas_profiling import ProfileReport


def plot_results(trials, task, algo, measure):
    f, ax = subplots(1)  # , figsize=(10,10))
    xs = [t['misc']['vals']['n_neighbors'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    xs = np.array(xs).squeeze()
    ys = np.array(ys).squeeze()
    ax.bar(xs, ys, width=0.5, linewidth=0.5, alpha=0.5)
    # ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
    ax.set_title(str(task) + ' - ' + str(algo), fontsize=18)
    ax.set_xlabel('n_neighbors', fontsize=12)
    ax.set_ylabel('cross validation accuracy', fontsize=12)
    print('saving plot...')
    savefig('training_loss_{}_{}_{}.png'.format(task, algo, measure))


def transform(df, aps, scaler, enc):
    # fit transform all the AP columns and target columns
    df[aps] = df[aps].fillna(100)
    df[['coord_x', 'coord_y', 'coord_z']] = df[['coord_x', 'coord_y', 'coord_z']].fillna(0)
    # fill categorical missing values with 'missing'
    df[['building', 'floor', 'tile']] = df[['building', 'floor', 'tile']].fillna('missing')
    df[['coord_x', 'coord_y', 'coord_z'] + aps] = scaler.fit_transform(df[['coord_x', 'coord_y', 'coord_z'] + aps])
    df[['building', 'floor', 'tile']] = enc.fit_transform(df[['building', 'floor', 'tile']])
    # drop the rows with missing values
    df = df.dropna()
    return df


# save function save the model
def save(model, task, algorithm, distance, aps):
    # Save the model
    print('Saving model...')
    # timestamp datetime
    date = datetime.now()
    ts = date.timestamp()
    ts = str(ts).split('.')[0]
    name_model_f = 'model_{}_{}_{}_{}'.format(task, algorithm, distance, ts)
    with open(name_model_f + '.pkl', 'wb') as f:
        # save aps as list of features in the model
        model.aps = aps
        pkl.dump(model, f)
    f.close()
    return name_model_f


# load function load the model
def load(model_path):
    # Load the model
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
    f.close()
    return model


# this function save csv results of the dataframes
def save_csv(final_res, metrics_df):
    final_res.to_csv('final_results.csv')
    metrics_df.to_csv('metrics.csv')


def save_excel(final_res, metrics_df):
    with pd.ExcelWriter('final_results.xlsx') as writer:
        final_res.to_excel(writer, sheet_name='final_results')
        metrics_df.to_excel(writer, sheet_name='metrics')


def features_check(data_test, aps_train):
    try:
        data_test = data_test[aps_train]
    except KeyError:
        print('data_test != aps_train')
    return data_test


# used to find the columns of the dataframe that contains the APs
def find_aps(df):
    columns = []
    for col in df.columns:
        if 'AP' in col:
            columns.append(col)
    return columns

