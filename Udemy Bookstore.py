"""
A program that stores this book information
Title, Author, Year, ISBN
User can view, search, add, update and delete entries

pyinstaller --onefile --windowed "Udemy Bookstore.py"
"""

from bookstorebackend import Database
database=Database("books.db")

from tkinter import *
window = Tk()
window.wm_title("Book Store")

title=Label(text='Title')
title.grid(row=0,column=0)
titleinput_value = StringVar()
titleinput=Entry(window,textvariable=titleinput_value)
titleinput.grid(row=0,column=1)

year=Label(text='Year')
year.grid(row=1,column=0)
yearinput_value = StringVar()
yearinput=Entry(window,textvariable=yearinput_value)
yearinput.grid(row=1,column=1)

author=Label(text='Author')
author.grid(row=0,column=2)
authorinput_value = StringVar()
authorinput=Entry(window,textvariable=authorinput_value)
authorinput.grid(row=0,column=3)


isbn=Label(text='ISBN')
isbn.grid(row=1,column=2)
isbninput_value = StringVar()
isbninput=Entry(window,textvariable=isbninput_value)
isbninput.grid(row=1,column=3)

def get_selected_row(event):
    global selected_tuple
    try:
        index=listbox.curselection()[0]
        selected_tuple=listbox.get(index)
        titleinput.delete(0,END)
        titleinput.insert(END,selected_tuple[1])
        authorinput.delete(0,END)
        authorinput.insert(END,selected_tuple[2])
        yearinput.delete(0,END)
        yearinput.insert(END,selected_tuple[3])
        isbninput.delete(0,END)
        isbninput.insert(END,selected_tuple[4])
    except IndexError:
        pass
    

def view_command():
    titleinput.delete(0,END)
    authorinput.delete(0,END)
    yearinput.delete(0,END)
    isbninput.delete(0,END)
    listbox.delete(0,END)
    for row in database.view():
        listbox.insert(END,row)

viewall=Button(window,text="View All",width=12,command=view_command)
viewall.grid(row=2,column=3) # or b1.pack()

def search_command():
    listbox.delete(0,END)
    for row in database.search(titleinput_value.get(),authorinput_value.get(),yearinput_value.get(),isbninput_value.get()):
        listbox.insert(END,row)


searchentry=Button(window,text="Search Entry",width=12,command=search_command)
searchentry.grid(row=3,column=3) # or b1.pack()

def insert_command():
    if "" not in [titleinput_value.get().strip(), authorinput_value.get().strip() , yearinput_value.get().strip() , isbninput_value.get().strip()]:
        database.insert(titleinput_value.get(),authorinput_value.get(),yearinput_value.get(),isbninput_value.get())
        view_command()
     
addentry=Button(window,text="Add Entry",width=12,command=insert_command)
addentry.grid(row=4,column=3) # or b1.pack()

def update_command():
    if "" not in [titleinput_value.get().strip(), authorinput_value.get().strip() , yearinput_value.get().strip() , isbninput_value.get().strip()]:
        database.update(selected_tuple[0],titleinput_value.get(),authorinput_value.get(),yearinput_value.get(),isbninput_value.get())
        view_command()

update=Button(window,text="Update",width=12,command=update_command)
update.grid(row=5,column=3) # or b1.pack()

def delete_command():
    database.delete(selected_tuple[0])
    view_command()
    

delete=Button(window,text="Delete",width=12,command=delete_command)
delete.grid(row=6,column=3) # or b1.pack()

close=Button(window,text="Close",width=12,command=window.destroy)
close.grid(row=7,column=3) # or b1.pack()

listbox=Listbox(window,height=6,width=35)
listbox.grid(row=2,column=0,rowspan=6,columnspan=2)
scroll=Scrollbar(window)
scroll.grid(row=2,column=2,rowspan=6)
listbox.configure(yscrollcommand=scroll.set)
scroll.configure(command=listbox.yview)
listbox.bind('<<ListboxSelect>>',get_selected_row)




window.mainloop()