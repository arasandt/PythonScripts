
# =============================================================================
# import re
# 
# line = "Cats are smarter than dogs"
# 
# matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)
# print(matchObj)
# if matchObj:
#    print("matchObj.group() : ", matchObj.group())
#    print("matchObj.group(1) : ", matchObj.group(1))
#    print("matchObj.group(2) : ", matchObj.group(2))
# else:
#    print ("No match!!")
#    
# =============================================================================
   
   
# =============================================================================
# import os
# 
# print ("Content-type: text/html\r\n\r\n")
# print ("<font size=+1>Environment</font><\br>")
# for param in os.environ.keys():
#    print ("<b>%20s</b>: %s<\br>" % (param, os.environ[param]))
# =============================================================================

# =============================================================================
# import threading
# import time
# 
# class myThread (threading.Thread):
#    def __init__(self, threadID, name, counter):
#       threading.Thread.__init__(self)
#       self.threadID = threadID
#       self.name = name
#       self.counter = counter
#       
#    def run(self):
#       print ("Starting " + self.name)
#       # Get lock to synchronize threads
#       threadLock.acquire()
#       print_time(self.name, self.counter, 3)
#       # Free lock to release next thread
#       threadLock.release()
# 
# def print_time(threadName, delay, counter):
#    while counter:
#       time.sleep(delay)
#       print( "%s: %s" % (threadName, time.ctime(time.time())))
#       counter -= 1
# 
# threadLock = threading.Lock()
# threads = []
# 
# # Create new threads
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)
# 
# # Start new Threads
# thread2.start()
# thread1.start()
# 
# 
# # Add threads to thread list
# threads.append(thread1)
# threads.append(thread2)
# 
# # Wait for all threads to complete
# for t in threads:
#     t.join()
# print ("Exiting Main Thread")
# =============================================================================



# =============================================================================
# #from tkinter import RAISED
# import tkinter
# from tkinter import messagebox
# 
# top = tkinter.Tk()
# 
# def helloCallBack():
#    messagebox.showinfo( "Hello Python", "Hello World")
# 
# 
# B = tkinter.Button(top, text ="Hello", command = helloCallBack)
# B.pack()
# top.mainloop()
# =============================================================================


# =============================================================================
# import pandas as pd
# 
# class DataUsage():
#     def __init__(self,customerid,usageDate,pulse,amount):
#         self.customerid = customerid
#         self.usageDate = usageDate
#         self.pulse = pulse
#         self.amount = amount
#         
# 
# #df = pd.read_csv('D:\Arasan\Misc\GitHub\Others\input\datausage.csv')
# df = pd.read_csv('.\\input\\datausage.csv')
# #print(df['Customer Id'])
# print(df[df['Customer Id'] == 'CUS1101314'])
# #print(df.head())
# =============================================================================

# =============================================================================
# #from xml.dom.minidom import parse
# import xml.dom.minidom
# import urllib.request
# #from urllib.request import Request, urlopen
# #data = urlopen("http://test.cognizant.e-box.co.in/uploads/call.xml").read().decode('ascii')
# data = urllib.request.urlretrieve("http://test.cognizant.e-box.co.in/uploads/call.xml")
# #print(data)
# DOMTree = xml.dom.minidom.parse(data[0])
# #collection = 
# datausage = DOMTree.documentElement.getElementsByTagName("datausage")
# #print(datausage)
# for i in datausage:
#     customerid = i.getElementsByTagName('customerid')[0]
#     tonumber = i.getElementsByTagName('tonumber')[0]
#     duration = i.getElementsByTagName('duration')[0]
#     amount = i.getElementsByTagName('amount')[0]
#     print(customerid.childNodes[0].data,tonumber.childNodes[0].data,duration.childNodes[0].data,amount.childNodes[0].data)
# 
# 
# 
# 
# 
# 
# =============================================================================
import datetime

a = '20-Aug-2018'
x = datetime.datetime.strptime(a,'%d-%b-%Y')
print(datetime.datetime.weekday(x))
"https://raw.githubusercontent.com/arasandt/LearnPython/master/input/datausage.csv"

























