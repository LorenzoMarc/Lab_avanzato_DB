''' this module evaluate the prediction and the dataset uploaded by the user '''

from tkinter import Button, Label, Toplevel
from tkinter.filedialog import askopenfile
from tkinter.ttk import Checkbutton
from predictor import main_test_pred
from tkinter import IntVar


def main():
    main_w = Toplevel()
    main_w.title('Lab2 Positional ML')
    main_w.geometry('500x400')

    # read and save the prediction .csv file uploaded by the user

    def select_prediction():
        global prediction
        file_path = askopenfile(mode='r', filetypes=[('csv', '*csv')])
        if file_path is not None:
            prediction = file_path.name
            Label(main_w, text="Prediction selected: \n" + str(prediction), foreground='green').grid(row=7,
                                                                                                     columnspan=3,
                                                                                                     pady=10)
        else:
            prediction = None

    def run_main():
        metrics = {'RMSE': rmse_bool.get(), 'Multi': multi_met_bool.get(), 'Accuracy': acc_bool.get()}
        if prediction is None:
            print("Select a valid prediction")
            Label(main_w, text="Select a valid prediction", foreground='red').grid(row=7, columnspan=3, pady=10)
        else:
            done = main_test_pred(prediction, metrics)
            # print res
            print("Result: \n" + str(done))

    # BUTTON upload your prediction
    btn_pred = Button(main_w, text='Upload your prediction',
                      command=select_prediction)
    btn_pred.grid(row=1, column=1)

    section_reg = Label(main_w, text='Regression metrics')
    section_reg.grid(row=2, column=1, padx=10, columnspan=2)

    # Metrics selection
    rmse_bool = IntVar()
    multi_met_bool = IntVar()
    acc_bool = IntVar()
    rmse_box = Checkbutton(main_w, text='RMSE',  variable=rmse_bool, onvalue=1, offvalue=0)
    rmse_box.grid(row=3, column=0)
    multi_met_box = Checkbutton(main_w, text='Multi', variable=multi_met_bool, onvalue=1,
                                offvalue=0)
    multi_met_box.grid(row=3, column=2, padx=10)

    section_clf = Label(main_w, text='Classification metrics')
    section_clf.grid(row=4, column=1, padx=10, columnspan=2)

    acc_box = Checkbutton(main_w, text='Accuracy', variable=acc_bool, onvalue=1,
                          offvalue=0)
    acc_box.grid(row=5, column=1)

    upld = Button(
        main_w,
        text='Compute Files',
        command=run_main
    )
    upld.grid(row=8, columnspan=3, pady=10)

    main_w.mainloop()
