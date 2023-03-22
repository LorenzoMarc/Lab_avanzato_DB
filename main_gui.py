from tkinter import *
from tkinter.filedialog import askopenfile
from tkinter.ttk import *
import predictor


def main():
    main_w = Tk()
    main_w.title('Lab2 Positional ML')
    main_w.geometry('500x400')

    OPTIONS = [
        "SELECT ALGORITHM",
        "KNN",
        "ALGO2"
    ]

    MODE = ['SELECT MODE', 'TRAIN', 'PREDICT']

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
            Label(main_w, text="model selected: \n" + str(upload_model), foreground='green').grid(row=11, columnspan=3,
                                                                                                  pady=10)
        else:
            print("Select a pickle file .pkl")
            Label(main_w, text="Select a model file .pkl", foreground='red').grid(row=10, columnspan=3, pady=10)

    def run_main():
        algo_selected = variable.get()
        mode_selected = variable_mode.get()
        if dataset is None:
            print("Select a valid Dataset")
            Label(main_w, text="Select a valid Dataset", foreground='red').grid(row=10, columnspan=3, pady=10)
        elif algo_selected is None:
            print("Select a valid ML algorithm")
            Label(main_w, text="Select a valid ML algorithm", foreground='red').grid(row=10, columnspan=3, pady=10)
        elif mode_selected is None:
            print("Select a valid mode")
            Label(main_w, text="Select a valid mode", foreground='red').grid(row=10, columnspan=3, pady=10)

        check_valid = any(algo_selected in algo for algo in OPTIONS[1:])
        if not check_valid:
            print("Select a valid algo")
            Label(main_w, text="Select a valid algo", foreground='red').grid(row=10, columnspan=3, pady=10)
        else:
            port = textport.get(1.0, "end-1c")
            db = textdb.get(1.0, "end-1c")
            user = textuser.get(1.0, "end-1c")
            psw = textpsw.get(1.0, "end-1c")
            host = texthost.get(1.0, "end-1c")

            if (port and db and user and psw and host) == "":
                port = "5432"
                db = "postgres"
                user = "postgres"
                psw = "admin"
                host = "localhost"

            if mode_selected == 'TRAIN':
                pred, score, name_model = predictor.main_train_multilabel(dataset, algo_selected)
                # print res and score
                print("Result: \n" + str(name_model))
                print("Score: \n" + str(score))
                Label(main_w, text="Result: \n" + str(name_model) +'.xlsx', foreground='green').grid(row=11, columnspan=3, pady=10)
                Label(main_w, text="Score: \n" + str(score), foreground='green').grid(row=12, columnspan=3, pady=10)
            else:
                if upload_model is not None:
                    res, score, final_tab = predictor.main_test(dataset, upload_model)
                    final_tab.to_excel('final_tab.xlsx')
                    # print res and score
                    print("Result: \n" + str(res))
                    print("Score: \n" + str(score))
                    Label(main_w, text="Results saves in: \n" + 'final_tab.xlsx', foreground='green').grid(row=12, columnspan=3,
                                                                                            pady=10)
                else:
                    # print error for upload model
                    Label(main_w, text="Missing model, upload a pkl model of KNN", foreground='red').grid(row=11,
                                                                                                          columnspan=3, pady=10)

    # BUTTON select dataset
    btn = Button(main_w, text='Select dataset', command=select_data)
    btn.grid(row=0, column=1)

    # BUTTON custom model
    btn_model = Button(main_w, text='Upload your pickle model', command=select_pickle)
    btn_model.grid(row=1, column=1)

    # LIST select mode

    modes = Label(main_w, text='Select mode')
    modes.grid(row=2, column=0, padx=10)

    variable_mode = StringVar(main_w)

    mode_s = OptionMenu(main_w, variable_mode, *MODE)

    mode_s.grid(row=2, columnspan=2, column=1)

    # LIST select algorithm
    algolab = Label(main_w, text='Select algorithm')
    algolab.grid(row=3, column=0, padx=10)

    variable = StringVar(main_w)

    algos = OptionMenu(main_w, variable, *OPTIONS)

    algos.grid(row=3, columnspan=2, column=1)

    # BUTTON compute

    upld = Button(
        main_w,
        text='Compute Files',
        command=run_main
    )
    upld.grid(row=4, columnspan=3, pady=10)

    # Connection Parameters UI
    # Porta db
    textport = Text(
        main_w, width=10, height=1
    )
    textport.grid(row=5, column=1)

    porttxt = Label(
        main_w,
        text='Select port db'
    )
    porttxt.grid(row=5, column=0, padx=10)

    # Nome DB
    textdb = Text(
        main_w, width=10, height=1
    )
    textdb.grid(row=6, column=1)

    dbtxt = Label(
        main_w,
        text='Select db'
    )
    dbtxt.grid(row=6, column=0, padx=10)

    #  username
    textuser = Text(
        main_w, width=10, height=1
    )
    textuser.grid(row=7, column=1)

    usertxt = Label(
        main_w,
        text='Select user db'
    )
    usertxt.grid(row=7, column=0, padx=10)

    #  psw
    textpsw = Text(
        main_w, width=10, height=1
    )
    textpsw.grid(row=8, column=1)

    pswtxt = Label(
        main_w,
        text='PSW DB'
    )
    pswtxt.grid(row=8, column=0, padx=10)

    #  host
    texthost = Text(
        main_w, width=10, height=1
    )
    texthost.grid(row=9, column=1)

    hosttxt = Label(
        main_w,
        text='Host DB'
    )
    hosttxt.grid(row=9, column=0, padx=10)

    main_w.mainloop()

if __name__ == '__main__':
    main()