from tkinter import *
import numpy as np
from multiprocessing import Process, Queue
from datetime import datetime
from queue import Empty
import cv2
#import cv2.cv as cv
from PIL import Image, ImageTk
import time
import tkinter as tk

#tkinter GUI functions----------------------------------------------------------
def quit_(root, process):
   process.terminate()
   root.destroy()

def update_image(image_label, queue):
   frame = queue.get()
   im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   a = Image.fromarray(im)
   b = ImageTk.PhotoImage(image=a)
   image_label.configure(image=b)
   image_label._image_cache = b  # avoid garbage collection
   root.update()

def update_label(display_label):
    display_label['text'] = str(datetime.now()) + "\n" + "is the time"
    #print(datetime.now())

def update_all(root, image_label, display_label, queue):
   update_image(image_label, queue)
   update_label(display_label)
   root.after(0, func=lambda: update_all(root, image_label, display_label, queue))

#multiprocessing image processing functions-------------------------------------
def image_capture(queue):
    vidFile = cv2.VideoCapture('a.mp4')
    while True:
        try:
            flag, frame=vidFile.read()
            if flag==0:
                break
            queue.put(frame)
            cv2.waitKey(20)
        except:
            continue

if __name__ == '__main__':
    queue = Queue()
    #print('queue initialized...')
    root = tk.Tk()
    #print('GUI initialized...')
    labelframe = Frame(root)
    #display_label.pack()   
    #image_label.pack()
    #print('GUI image label initialized...')
    p = Process(target=image_capture, args=(queue,))
    p.start()
    #print('image capture process has started...')
    # quit button

    image_label = tk.Label(labelframe, width=800)# label for the video frame
    display_label = tk.Label(labelframe, width=25)
    quit_button = tk.Button(labelframe, text='Quit',command=lambda: quit_(root,p))
    
    image_label.grid(row=1, column=1, sticky=E)
    display_label.grid(row=3, column=1, sticky=E)
    quit_button.grid(row=2, column=1, sticky=E)
    labelframe.pack(fill=X)

    
    #quit_button.pack()
    
    #print('quit button initialized...')
    # setup the update callback
    root.after(0, func=lambda: update_all(root, image_label, display_label, queue))
    #print('root.after was called...')
    root.mainloop()
    #print('mainloop exit')
    p.join()
    #print('image capture process exit')