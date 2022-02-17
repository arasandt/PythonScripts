import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk


root = tk.Tk()
#fname = "img1.png"

#img = Image.open(fname)
#photo = ImageTk.PhotoImage(img)

data = [['a123123','b12312312','c12312312','d1231231','img1.png'],
        ['1123123','223423423','323423432','41231231','img2.png'],
        ['!2342143234','@234234','#234234','$1231231','img3.png']]
        
for i,j in enumerate(data):
    (a,b,c,d,e) = j
    img = Image.open(e)
    img = img.resize((64,64), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    data[i][4] = photo
    
topbar = Frame(root)
         
for i,d in enumerate(data):
    (time, id, mess, secc, pho) = d
    #print(i, time, id, mess, secc, pho)
    
    labelframe = Frame(topbar)
    testlabel1 = Label(labelframe,text=time, width=25)
    testlabel2 = Label(labelframe,text=id, width=25)
    testlabel3 = Label(labelframe,text=mess, width=25)
    testlabel4 = Label(labelframe,text=secc, width=25)
    testlabel5 = Label(labelframe,image=pho)

    testlabel1.grid(row=i, column=1, sticky=E)
    testlabel2.grid(row=i, column=2, sticky=E)
    testlabel3.grid(row=i, column=3, sticky=E)
    testlabel4.grid(row=i, column=4, sticky=E)
    testlabel5.grid(row=i, column=0, sticky=E)

    labelframe.pack(fill=X)

topbar.pack(fill=X)

tk.mainloop()

#import sys
#import tkinter as tk
#from tkinter import ttk, Listbox
#from tkinter import *
#from PIL import Image,ImageTk,ImageFilter,ImageOps
#
#
#global fname
#fname = "img.png"
#
#data = [['a','b','c'],['1','2','3'],['!','@','#']]
#
#def browse_file():
#    fname = tk.filedialog.askopenfilename(filetypes=(("Bitmap files", "*.bmp"), ("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*")))
#    print(fname)
#    return
#
#def classify_obj():
#    print("In Development")
#    return
#
#
#root = tk.Tk()
#root.wm_title("Classify Image")
#
#count = len(data)
#frame1 = ttk.Frame(root, width=count * 100, height=count * 80)
#frame1.grid(row=0, column=0)
#frame1.columnconfigure(0, weight=1)
#frame1.rowconfigure(0, weight=1)
#list1 = Listbox(root, bd=0)
#list1.grid(column=0, row=0)
#
#list2 = Listbox(root, bd=0)
#list2.grid(column=1, row=0)
#
#
#separator1 = Frame(list1, height=2, bd=1, relief=SUNKEN)
#separator1.grid(column=0, row=0)
#separator2 = Frame(list1, height=2, bd=1, relief=SUNKEN)
#separator2.grid(column=1, row=0)
#separator3 = Frame(list2, height=2, bd=1, relief=SUNKEN)
#separator3.grid(column=2, row=0)
#separator4 = Frame(list2, height=2, bd=1, relief=SUNKEN)
#separator4.grid(column=3, row=0)
#e1 = Label(separator1, text='Label1')
#e1.grid(sticky=W+E)
#e2= Label(separator2, text='Label2')
#e2.grid(sticky=W+E)
#e3 = Label(separator3, text='Label3')
#e3.grid(sticky=W+E)
#e4= Label(separator4, text='Label4')

#im = Image.open(fname)
#photo = ImageTk.PhotoImage(im)
#cv = tk.Canvas(frame1, height=390, width=490, background="white", bd=1, relief=tk.RAISED)
#cv.grid(row=0,column=0)
#cv.create_image(0, 0, image=photo, anchor='nw')


#frame2 = tk.Frame(root, width=500, height=400, bd=1).grid(row=1, column=1)
#cv = tk.Canvas(frame2, height=390, width=490, bd=2, relief=tk.SUNKEN).grid(row=1,column=1)

#tk.mainloop()