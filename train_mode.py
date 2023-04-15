import glob
import os
from tkinter import *
from tkinter.filedialog import askopenfile
from tkinter.ttk import *
import pandas as pd
import predictor


# this function find folder from current path named 'converted dataset' and
# read names of subfolders and save them in a list
def get_folders():
    path = os.path.join(os.getcwd(), 'datasets')
    folders = os.listdir(path)
    return folders


# merge file fingerprints.csv and wifi_obs.csv in a single dataframe
def merge_fingerprints_wifi_obs(path_folder):
    df = pd.DataFrame()
    list_fingerprints = sorted(glob.glob(path_folder + r'\**\fingerprints.csv', recursive=True))
    list_wifi_obs = sorted(glob.glob(path_folder + r'\**\wifi_obs.csv', recursive=True))
    for f, w in zip(list_fingerprints, list_wifi_obs):
        if f.split('\\')[-2] != w.split('\\')[-2]:
            print('Error: fingerprints.csv and wifi_obs.csv are not in the same folder')
            return None
        else:
            try:
                # merge fingerprints.csv and wifi_obs.csv
                df_fingerprints = pd.read_csv(f)
                df_wifi_obs = pd.read_csv(w)
                df_files = pd.merge(df_fingerprints, df_wifi_obs, on='fingerprint_id')
                df = pd.concat([df, df_files])
            except:
                print('Error: ' + f + ' or ' + w + ' not found')
                continue
    return df


def main():
    ws = Toplevel()
    ws.title('Train Mode')
    ws.geometry('600x300')
    OPTIONS = [
        "SELECT ALGORITHM",
        "KNN",
        "WKNN"
    ]

    metrics_distance = [
        "SELECT METRICS",
        "cityblock",
        "euclidean",
        "l1",
        "l2",
        "manhattan",
        "haversine",
        "cosine"
    ]
    list_datasets = get_folders()
    list_datasets.insert(0, "SELECT DATASET")

    def select_data():
        global dataset
        file_path = askopenfile(mode='r', filetypes=[('csv', '*csv')])
        if file_path is not None:
            dataset = file_path.name
            Label(ws, text="Dataset selected: \n" + str(dataset), foreground='green').grid(row=10, columnspan=3,
                                                                                           pady=10)
        else:
            print("Select a dataset file .csv")
            Label(ws, text="Select a dataset file .csv", foreground='red').grid(row=10, columnspan=3, pady=10)

    def run_main():
        try:
            global training_set
            algo_selected = variable.get()
            distance_selected = variable_metric.get()
            n_neigh = int(num_neighbors.get())
            standard_dataset = list_name.get()

            if dataset is None and standard_dataset is None:
                print("Select a valid Dataset")
                Label(ws, text="Select a valid Dataset", foreground='red').grid(row=10, columnspan=3, pady=10)
            elif dataset is not None:
                training_set = pd.read_csv(dataset, low_memory=False)
            elif standard_dataset is not None:
                path_folder = os.path.join(os.getcwd(), 'datasets', standard_dataset)
                training_set = merge_fingerprints_wifi_obs(path_folder)
            elif algo_selected is None:
                print("Select a valid ML algorithm")
                Label(ws, text="Select a valid ML algorithm", foreground='red').grid(row=10, columnspan=3, pady=10)

            check_valid = any(algo_selected in algo for algo in OPTIONS[1:])
            check_valid_metric = any(distance_selected in metric for metric in metrics_distance[1:])
            if not check_valid:
                print("Select a valid algo")
                Label(ws, text="Select a valid algo", foreground='red').grid(row=10, columnspan=3, pady=10)

            elif not check_valid_metric:
                print("Select a valid metric")
                Label(ws, text="Select a valid metric", foreground='red').grid(row=10, columnspan=3, pady=10)
            else:
                res_class, res_reg = predictor.main_train_multilabel(training_set, algo_selected, distance_selected,
                                                                     n_neigh)
                # print res and score
                name_model = res_class[0]
                score = res_class[1]

                print("Result: \n" + str(name_model))
                print("Score: \n" + str(score))
                Label(ws, text="Result: \n" + str(name_model) + '.xlsx', foreground='green').grid(row=11, columnspan=3,
                                                                                                  pady=10)
                Label(ws, text="Score: \n" + str(score), foreground='green').grid(row=12, columnspan=3, pady=10)

        except Exception as e:
            # print error for upload model
            Label(ws, text="Error \n" + str(e), foreground='red').grid(row=11, columnspan=3, pady=10)

    # LIST standard datasets
    datasets_iso = Label(ws, text='Standard Dataset')
    datasets_iso.grid(row=0, column=0, padx=10)
    list_name = StringVar(ws)

    ds = OptionMenu(ws, list_name, *list_datasets)

    ds.grid(row=0, columnspan=2, column=1)

    # BUTTON select dataset
    btn = Button(ws, text='Select your dataset', command=select_data)
    btn.grid(row=1, column=1)

    # LIST select algorithm
    algolab = Label(ws, text='Select algorithm')
    algolab.grid(row=3, column=0, padx=10)

    variable = StringVar(ws)

    algos = OptionMenu(ws, variable, *OPTIONS)

    algos.grid(row=3, columnspan=2, column=1)

    # num Neighbors to insert
    num_neighbors = Label(ws, text='Number of Neighbors')
    num_neighbors.grid(row=3, column=4, padx=10)
    num_neighbors = Entry(ws)
    num_neighbors.grid(row=3, column=5)

    # LIST select algorithm
    metric_list = Label(ws, text='Select metrics distance')
    metric_list.grid(row=4, column=0, padx=10)

    variable_metric = StringVar(ws)

    metrics = OptionMenu(ws, variable_metric, *metrics_distance)

    metrics.grid(row=4, columnspan=2, column=1)

    # BUTTON compute

    upld = Button(
        ws,
        text='Compute Files',
        command=run_main
    )
    upld.grid(row=6, columnspan=3, pady=10, column=2)

    ws.mainloop()
