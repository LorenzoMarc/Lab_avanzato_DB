import pickle as pkl
from datetime import datetime
import numpy as np
from matplotlib.pyplot import savefig, subplots
import pandas as pd


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


# if the data_test has different columns from the aps_train, this function will padd the missing columns with 0
# if the data_test has more columns than the aps_train, this function will drop the extra columns
def features_check(data_test, aps_train):
    # extract from data_test the columns in aps_train
    data_test = data_test[aps_train]
    return data_test


# used to find the columns of the dataframe that contains the APs
def find_aps(df):
    columns = []
    for col in df.columns:
        if 'AP' in col:
            columns.append(col)
    return columns


# rename the features with incremental numbers
def rename_aps(df_train):
    aps = find_aps(df_train)
    aps = sorted(aps)
    for i, ap in enumerate(aps):
        df_train = df_train.rename(columns={ap: 'AP' + str(i)})
    return df_train
