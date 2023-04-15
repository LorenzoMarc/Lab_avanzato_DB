from tkinter import *
from tkinter.filedialog import askopenfile
from tkinter.ttk import *
import predictor


def main():
    main_w = Tk()
    main_w.title('Lab2 Positional ML')
    main_w.geometry('500x400')
    metric_list = []
    OPTIONS = [
        "SELECT ALGORITHM",
        "KNN",
        "WKNN"
    ]

    TASK = ['SELECT TASK', 'REGRESSION', 'CLASSIFICATION']

    def select_data():
        global dataset
        file_path = askopenfile(mode='r', filetypes=[('csv', '*csv')])
        if file_path is not None:
            dataset = file_path.name
            Label(main_w, text="Dataset selected: \n" + str(dataset), foreground='green').grid(row=10, columnspan=3,
                                                                                               pady=10)
        else:
            print("Select a dataset file .csv")
            Label(main_w, text="Select a dataset file .csv", foreground='red').grid(row=10, columnspan=3, pady=10)

    def select_pickle():
        global upload_model
        file_path = askopenfile(mode='r', filetypes=[('pkl', '*pkl')])
        if file_path is not None:
            upload_model = file_path.name
            Label(main_w, text="Model selected: \n" + str(upload_model), foreground='green').grid(row=11, columnspan=3,
                                                                                                  pady=10)
        else:
            print("Select a pickle file .pkl")
            Label(main_w, text="Select a model file .pkl", foreground='red').grid(row=10, columnspan=3, pady=10)

    def run_main():
        task = task_selected.get()

        if dataset is None:
            print("Select a valid Dataset")
            Label(main_w, text="Select a valid Dataset", foreground='red').grid(row=10, columnspan=3, pady=10)
        else:

            if upload_model is not None:
                res, final_tab = predictor.main_test(dataset, upload_model, metric_list, task)
                # print res
                print("Result: \n" + str(res))
                Label(main_w, text="Results saves in: \n" + 'final_tab.xlsx', foreground='green').grid(row=12,
                                                                                                       columnspan=3,
                                                                                                       pady=10)
            else:
                # print error for upload model
                Label(main_w, text="Missing model, upload a pkl model of KNN", foreground='red').grid(row=11,
                                                                                                      columnspan=3,
                                                                                                      pady=10)
    # LIST select task
    task_label = Label(main_w, text='Select task')
    task_label.grid(row=0, column=0, padx=10)

    task_selected = StringVar(main_w)

    tasks = OptionMenu(main_w, task_selected, *TASK)

    tasks.grid(row=0, columnspan=1, column=1)

    # BUTTON custom model
    btn_model = Button(main_w, text='Upload your pickle model', command=select_pickle)
    btn_model.grid(row=1, column=1)

    # BUTTON select dataset
    btn = Button(main_w, text='Select dataset', command=select_data)
    btn.grid(row=2, column=1)

    section_reg = Label(main_w, text='Regression metrics')
    section_reg.grid(row=4, column=1, padx=10, columnspan=2)

    # Metrics selection
    rmse_box = Checkbutton(main_w, text='RMSE', command=metric_list.append('RMSE'))
    rmse_box.grid(row=5, column=0)
    multi_met_box = Checkbutton(main_w, text='Multi', command=metric_list.append('Multi'))
    multi_met_box.grid(row=5, column=2, padx=10)

    section_clf = Label(main_w, text='Classification metrics')
    section_clf.grid(row=6, column=1, padx=10, columnspan=2)

    acc_box = Checkbutton(main_w, text='accuracy', command=metric_list.append('Accuracy'))
    acc_box.grid(row=7, column=1)

    upld = Button(
        main_w,
        text='Compute Files',
        command=run_main
    )
    upld.grid(row=8, columnspan=3, pady=10)

    main_w.mainloop()






