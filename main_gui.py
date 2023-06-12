"""
this is the main UI for the project. It makes available
the 3 modes of the project: train, evaluate and evaluate prediction
"""

from tkinter import Button, Tk, CENTER
from train_mode import main as train
from eval_mode import main as eval
from eval_pred_mode import main as eval_pred

main_w = Tk()
main_w.title('Lab2 Positional ML')
main_w.geometry('350x250')


# TODO:
## aggiungere descrizione parametri di default


def run_train():
    train()


def run_eval():
    eval()


def run_eval_pred():
    eval_pred()


# BUTTON train mode
btn = Button(main_w, text='TRAIN', command=run_train)
btn.place(relx=0.5, rely=0.3, anchor=CENTER)

btn_model = Button(main_w, text='EVALUATE', command=run_eval)
btn_model.place(relx=0.5, rely=0.6, anchor=CENTER)

# BUTTON evaluate prediction
btn_pred = Button(main_w, text='EVALUATE PREDICTION', command=run_eval_pred)
btn_pred.place(relx=0.5, rely=0.9, anchor=CENTER)
main_w.mainloop()
