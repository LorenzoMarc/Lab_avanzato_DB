import glob
import os
import pandas as pd
import predictor
from tkinter import Button, Label, StringVar, Toplevel, Checkbutton, IntVar, W
from tkinter.filedialog import askopenfile
from tkinter.ttk import OptionMenu, Entry


# this function find folder from current path named 'datasets' and
# read names of subfolders and save them in a list to present in GUI
def get_folders():
    path = os.path.join(os.getcwd(), 'datasets')
    folders = os.listdir(path)
    return folders


# merge file from Indoor Positioning Standard to a single dataframe. File merged: fingerprints.csv and wifi_obs.csv
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
                df_fingerprints = pd.read_csv(f)
                df_wifi_obs = pd.read_csv(w)
                df_files = pd.merge(df_fingerprints, df_wifi_obs, on='fingerprint_id')
                df = pd.concat([df, df_files])
            except FileNotFoundError:
                print('Error: ' + f + ' or ' + w + ' not found')
                continue
    return df


standard_dataset = ''


def main():
    ws = Toplevel()
    ws.title('Train Mode')
    ws.geometry('780x300')

    # if you add a new algorithm you have to add it to this list to make it selectable
    OPTIONS = [
        "SELECT ALGORITHM",
        "KNN",
        "WKNN"
    ]

    # If you want to add a new measure of distance, add it to this list
    measure_distance = [
        "SELECT METRICS",
        "cityblock",
        "euclidean",
        "manhattan",
        "cosine"
    ]

    list_datasets = get_folders()
    list_datasets.insert(0, "SELECT STANDARD DATASET")

    def select_data():
        global standard_dataset
        path = askopenfile(mode='r', filetypes=[('csv', '*csv')])
        try:
            standard_dataset = path.name
            Label(ws, text="Dataset selected: \n" + str(standard_dataset), foreground='green').grid(
                row=10, columnspan=3, pady=10)
        except AttributeError:
            print('No file selected')
            Label(ws, text="No file selected", foreground='red').grid(row=10, columnspan=3, pady=10)

    def run_main():

        fine_tune = var.get()
        num_eval_selected = num_eval.get()
        algo_selected = variable.get()
        measure_selected = variable_measure.get()
        n_neigh = int(num_neighbors.get())
        dataset_sel = list_name.get()

        try:
            test_size = float(test_size_text.get())
        except:
            test_size = 0.2

        # checking input ...
        training_set = 'None training set selected'
        if dataset_sel in list_datasets[1:]:
            path_folder = os.path.join(os.getcwd(), 'datasets', dataset_sel)
            training_set = merge_fingerprints_wifi_obs(path_folder)
        elif standard_dataset is not None:
            training_set = pd.read_csv(standard_dataset, low_memory=False)

        elif algo_selected is None:
            print("Select a valid ML algorithm")
            Label(ws, text="Select a valid ML algorithm", foreground='red').grid(row=10, columnspan=3, pady=10)

        check_valid = any(algo_selected in algo for algo in OPTIONS[1:])
        check_valid_measure = any(measure_selected in measure for measure in measure_distance[1:])
        if not check_valid:
            print("Select a valid algo")
            Label(ws, text="Select a valid algo", foreground='red').grid(row=10, columnspan=3, pady=10)

        elif not check_valid_measure:
            print("Select a valid measure")
            Label(ws, text="Select a valid measure", foreground='red').grid(row=10, columnspan=3, pady=10)
        else:

            # everything is ok, run main training function --> follow this
            res_class, res_reg = predictor.main_train_multilabel(training_set, algo_selected, measure_selected,
                                                                 fine_tune, num_eval_selected, n_neigh, test_size)
            # print res and score of the first model
            result_model = res_class[0]
            score = res_class[1]
            # print the parameters selected for the training
            print("Parameters selected: \n")
            print("Algorithm: " + str(algo_selected))
            print("Measure: " + str(measure_selected))
            print("Fine tune: " + str(fine_tune))
            print("Number of evaluations: " + str(num_eval_selected))
            print("Number of neighbors: " + str(n_neigh))
            print("Test size: " + str(test_size))
            print("Result: \n" + str(result_model))
            print("Score: \n" + str(score))
            Label(ws, text="Result: \n" + str(result_model) + '.xlsx', foreground='green').grid(row=11, columnspan=3,
                                                                                              pady=10)
            Label(ws, text="Score: \n" + str(score), foreground='green').grid(row=12, columnspan=3, pady=10)
            # save the parameters selected for the training in xlsx file
            df = pd.DataFrame({'Algorithm': [algo_selected],
                                 'Measure': [measure_selected],
                                    'Fine tune': [bool(fine_tune)],
                                    'Number of evaluations': [num_eval_selected],
                                    'Number of neighbors': [n_neigh],
                                    'Test size': [test_size],
                                    'Score': [score]})
            df.to_csv(str(algo_selected) + '_param.csv', index=False)

    # LIST standard datasets
    datasets_iso = Label(ws, text='Standard Dataset')
    datasets_iso.grid(row=0, column=0)
    list_name = StringVar(ws)

    ds = OptionMenu(ws, list_name, *list_datasets)

    ds.grid(row=0, column=1)

    # BUTTON select dataset

    Label(ws, text='OR').grid(row=0, column=2)
    btn = Button(ws, text='Select your dataset', command=select_data)
    btn.grid(row=0, column=3)

    # LIST select algorithm
    algolab = Label(ws, text='Select algorithm')
    algolab.grid(row=1, column=0)

    variable = StringVar(ws)

    algos = OptionMenu(ws, variable, *OPTIONS)

    algos.grid(row=1, column=1)

    # num Neighbors to insert
    num_neighbors = Label(ws, text='Number of Max Neighbors')
    num_neighbors.grid(row=1, column=2)
    num_neighbors = Entry(ws)
    num_neighbors.grid(row=1, column=3)

    # checkbox to select if you want to finetune the model or not
    var = IntVar()
    Checkbutton(ws, text="Fine tune model", variable=var, offvalue=False, onvalue=True).grid(row=2, column=2)

    # var num evaluation
    Label(ws, text='Max evaluation tentatives of the optimizer').grid(row=2, column=0)
    num_eval = Entry(ws)
    num_eval.grid(row=2, column=1)

    # var num test_size
    Label(ws, text='Test Size for training \n Only decimal format allowed (0.2 default)').grid(row=4, column=0)
    test_size_text = Entry(ws)
    test_size_text.grid(row=4, column=1)

    # LIST select algorithm
    measure_list = Label(ws, text='Select measure distance for KNN')
    measure_list.grid(row=3, column=0)

    variable_measure = StringVar(ws)

    metrics = OptionMenu(ws, variable_measure, *measure_distance)

    metrics.grid(row=3,  column=1)

    # BUTTON compute

    upld = Button(
        ws,
        text='Compute Files',
        command=run_main
    )
    upld.grid(row=5, column=2)

    ws.mainloop()
