''' this module evaluate the prediction and the dataset uploaded by the user '''

from tkinter import Button, Label, Toplevel
from tkinter.filedialog import askopenfile
from tkinter.ttk import Checkbutton
from predictor import main_test_pred
from tkinter import IntVar


def main():
    main_w = Toplevel()
    main_w.title('Your Prediction Evaluation')
    main_w.geometry('500x400')

    # read and save the prediction .csv file uploaded by the user

    def select_prediction():
        global prediction
        file_path = askopenfile(mode='r', filetypes=[('csv', '*csv')])
        if file_path is not None:
            prediction = file_path.name
            Label(main_w, text="Prediction selected: \n" + str(prediction), foreground='green').grid(row=12,
                                                                                                     columnspan=3,
                                                                                                     pady=10)
        else:
            prediction = None

    def run_main():
        metrics = { 'RMSE': rmse_bool.get(), 'Precision': precision_bool.get(), 'Recall': recall_bool.get(),
                    'F1': f1_bool.get(), 'MAE': mae_bool.get(), 'MSE': mse_bool.get(), 'ME': me_bool.get(),
                     'R2': r2_bool.get(), 'EVS': evs_bool.get(), 'MedAE': medae_bool.get(),
                    'Accuracy': acc_bool.get(), '2D Error': error2d_bool.get(), 'fdp': fdp_bool.get()}
        if prediction is None:
            print("Select a valid prediction")
            Label(main_w, text="Select a valid prediction", foreground='red').grid(row=12, columnspan=3, pady=10)
        else:
            done = main_test_pred(prediction, metrics)
            # print res
            if done:
                print("Result avaialble in metrics.csv")
            else:
                print("Error in the evaluation")

    # BUTTON upload your prediction
    btn_pred = Button(main_w, text='Upload your prediction',
                      command=select_prediction)
    btn_pred.grid(row=1, column=1)

    section_reg = Label(main_w, text='Regression metrics')
    section_reg.grid(row=2, column=1, padx=10, columnspan=2)

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

    # Create checkbox for each metric
    rmse_box = Checkbutton(main_w, text='RMSE', variable=rmse_bool, onvalue=1, offvalue=0, )
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

    section_clf = Label(main_w, text='Classification metrics')
    section_clf.grid(row=8, column=1, padx=10, columnspan=2)

    acc_box = Checkbutton(main_w, text='accuracy', variable=acc_bool, onvalue=1, offvalue=0)
    acc_box.grid(row=9, column=1)

    upld = Button(
        main_w,
        text='Compute Files',
        command=run_main
    )
    upld.grid(row=11, columnspan=3, pady=10)

    main_w.mainloop()
