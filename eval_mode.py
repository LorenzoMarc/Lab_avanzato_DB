from tkinter import Button, Label, StringVar, Toplevel, IntVar
from tkinter.filedialog import askopenfile
from tkinter.ttk import Checkbutton, OptionMenu
from predictor import main_test


def main():
    main_w = Toplevel()
    main_w.title('Algo Evaluation')
    main_w.geometry('800x400')

    TASK = ['SELECT TASK', 'REGRESSION', 'CLASSIFICATION']

    def select_data():
        global dataset
        file_path = askopenfile(mode='r', filetypes=[('csv', '*csv')])
        if file_path is not None:
            dataset = file_path.name
            Label(main_w, text="Dataset selected: \n" + str(dataset), foreground='green').grid(row=13, columnspan=3,
                                                                                               pady=10)
        else:
            print("Select a dataset file .csv")
            Label(main_w, text="Select a dataset file .csv", foreground='red').grid(row=13, columnspan=3, pady=10)

    def select_pickle():
        global upload_model
        file_path = askopenfile(mode='r', filetypes=[('pkl', '*pkl')])
        if file_path is not None:
            upload_model = file_path.name
            Label(main_w, text="Model selected: \n" + str(upload_model), foreground='green').grid(row=13, columnspan=3,
                                                                                                  pady=10)
        else:
            print("Select a pickle file .pkl")
            Label(main_w, text="Select a model file .pkl", foreground='red').grid(row=13, columnspan=3, pady=10)

    def run_main():
        task = task_selected.get()
        # add in metrics dictionary the metrics you want to implement
        metrics = { 'RMSE': rmse_bool.get(), 'Precision': precision_bool.get(), 'Recall': recall_bool.get(),
                    'F1': f1_bool.get(), 'MAE': mae_bool.get(), 'MSE': mse_bool.get(), 'ME': me_bool.get(),
                     'R2': r2_bool.get(), 'EVS': evs_bool.get(), 'MedAE': medae_bool.get(),
                    'Accuracy': acc_bool.get(), '2D Error': error2d_bool.get(), 'fdp': fdp_bool.get(),
                    'SD': sd_bool.get(), 'EVAAL': evaal_bool.get()}

        if dataset is None:
            print("Select a valid Dataset")
            Label(main_w, text="Select a valid Dataset", foreground='red').grid(row=13, columnspan=3, pady=10)
        else:

            if upload_model is not None:
                print('model selected: ', upload_model)
                res, final_tab = main_test(dataset, upload_model, metrics, task)
                # print res
                print("Result: \n" + str(res))
                Label(main_w, text="Results saves in: \n" + 'final_tab.xlsx', foreground='green').grid(row=13,
                                                                                                       columnspan=3,
                                                                                                       pady=10)
            else:
                # print error for upload model
                Label(main_w, text="Missing model, upload a pkl model of KNN", foreground='red').grid(row=13,
                                                                                                      columnspan=3,
                                                                                                      pady=10)
    # LIST select task
    task_label = Label(main_w, text='Select task')
    task_label.grid(row=0, column=0, padx=10)

    task_selected = StringVar(main_w)

    tasks = OptionMenu(main_w, task_selected, *TASK)

    tasks.grid(row=0, columnspan=1, column=1)

    # BUTTON custom model
    Label(main_w, text='Upload trained model').grid(row=1, column=0, padx=10)
    btn_model = Button(main_w, text='Upload your pickle model', command=select_pickle)
    btn_model.grid(row=1, column=1)

    # BUTTON select dataset
    Label(main_w, text='Dataset compliant to Indoor Positioning DB').grid(row=2, column=0, padx=10)
    btn = Button(main_w, text='Select dataset', command=select_data)
    btn.grid(row=2, column=1)

    section_reg = Label(main_w, text='Regression metrics')
    section_reg.grid(row=4, column=1, padx=10, columnspan=2)

    # Metrics selection
    # add IntVar for the metrics precision, recall, f1, accuracy, rmse, mae, mape, r2
    rmse_bool = IntVar()
    acc_bool = IntVar()
    precision_bool = IntVar()
    recall_bool = IntVar()
    f1_bool = IntVar()
    mae_bool = IntVar()
    me_bool = IntVar()
    r2_bool = IntVar()
    mse_bool = IntVar()
    evs_bool = IntVar()
    medae_bool = IntVar()
    error2d_bool = IntVar()
    fdp_bool = IntVar()
    sd_bool = IntVar()
    evaal_bool = IntVar()

    # Create checkbox for each metric
    rmse_box = Checkbutton(main_w, text='RMSE', variable=rmse_bool, onvalue=1, offvalue=0,)
    rmse_box.grid(row=5, column=0)
    mse_box = Checkbutton(main_w, text='MSE', variable=mse_bool, onvalue=1, offvalue=0)
    mse_box.grid(row=5, column=1)
    vse_box = Checkbutton(main_w, text='VSE', variable=evs_bool, onvalue=1, offvalue=0)
    vse_box.grid(row=5, column=2)
    medae_box = Checkbutton(main_w, text='MedAE', variable=medae_bool, onvalue=1, offvalue=0)
    medae_box.grid(row=5, column=3)
    precision_box = Checkbutton(main_w, text='Precision', variable=precision_bool, onvalue=1, offvalue=0)
    precision_box.grid(row=6, column=0)
    recall_box = Checkbutton(main_w, text='Recall', variable=recall_bool, onvalue=1, offvalue=0)
    recall_box.grid(row=6, column=1)
    f1_box = Checkbutton(main_w, text='F1', variable=f1_bool, onvalue=1, offvalue=0)
    f1_box.grid(row=6, column=2)
    mae_box = Checkbutton(main_w, text='MAE', variable=mae_bool, onvalue=1, offvalue=0)
    mae_box.grid(row=7, column=0)
    me_box = Checkbutton(main_w, text='ME', variable=me_bool, onvalue=1, offvalue=0)
    me_box.grid(row=7, column=1)
    r2_box = Checkbutton(main_w, text='R2', variable=r2_bool, onvalue=1, offvalue=0)
    r2_box.grid(row=7, column=2)
    error2d_box = Checkbutton(main_w, text='2D Error', variable=error2d_bool, onvalue=1, offvalue=0)
    error2d_box.grid(row=7, column=3)
    fdp_box = Checkbutton(main_w, text='Floor Detection Rate', variable=fdp_bool, onvalue=1, offvalue=0)
    fdp_box.grid(row=6, column=3)
    sd_box = Checkbutton(main_w, text='Standard Deviation', variable=sd_bool, onvalue=1, offvalue=0)
    sd_box.grid(row=8, column=0)
    evaal_box = Checkbutton(main_w, text='EVAAL', variable=evaal_bool, onvalue=1, offvalue=0)
    evaal_box.grid(row=8, column=1)

    section_clf = Label(main_w, text='Classification metrics')
    section_clf.grid(row=9, column=1, padx=10, columnspan=2)

    acc_box = Checkbutton(main_w, text='accuracy', variable=acc_bool, onvalue=1, offvalue=0)
    acc_box.grid(row=10, column=1)


    upld = Button(
        main_w,
        text='Compute Files',
        command=run_main
    )
    upld.grid(row=12, columnspan=3, pady=10)

    main_w.mainloop()
