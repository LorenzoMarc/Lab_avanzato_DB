from tkinter import *
from tkinter.ttk import *
import train_mode, eval_mode

main_w = Tk()
main_w.title('Lab2 Positional ML')
main_w.geometry('350x250')


def run_train():
    train_mode.main()


def run_eval():
    eval_mode.main()


# BUTTON train mode
btn = Button(main_w, text='TRAIN', command=run_train)
btn.place(relx=0.5, rely=0.3, anchor=CENTER)

btn_model = Button(main_w, text='EVALUATE', command=run_eval)
btn_model.place(relx=0.5, rely=0.6, anchor=CENTER)

main_w.mainloop()
