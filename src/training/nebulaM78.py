#/usr/bin/env/python
# a set of useless functions;
# created by nebulaM78 team; capstone 2020;
# src - http://bluegalaxy.info/codewalk/2017/10/14/python-how-to-create-gui-pop-up-windows-with-tkinter/
# src - https://pythonprogramming.net/tkinter-popup-message-window/
# src - https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms%20old/Demo_Threaded_Work.py

from Tkinter import *

def alert_popup(title, message, path):
    """Generate a pop-up window for special messages."""
    root = Tk()
    root.title(title)
    w = 400     # popup window width
    h = 200     # popup window height
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw - w)/2
    y = (sh - h)/2
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    m = message
    m += '\n'
    m += path
    w = Label(root, text=m, width=120, height=10)
    w.pack()
    b = Button(root, text="OK", command=root.destroy, width=10)
    b.pack()
    mainloop()

# test driver;
if __name__ == '__main__':
    alert_popup("Title goes here..", "Hello World!", "A path or second message can go here..")
   