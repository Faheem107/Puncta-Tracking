import tkinter as tk
from tkinter import filedialog, simpledialog

def _gui(_func,*args,**kwargs):
    root = tk.Tk()
    root.withdraw()
    return _func(*args,**kwargs)

def get_path(*args,**kwargs):
	return _gui(filedialog.askopenfilename,*args,**kwargs)

def get_directory(*args,**kwargs):
    return _gui(filedialog.askdirectory,*args,**kwargs)

def get_int(*args,**kwargs):
    return _gui(simpledialog.askinteger,*args,**kwargs)
