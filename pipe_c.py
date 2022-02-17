import time
import win32pipe, win32file, pywintypes
import tkinter as tk
from datetime import timedelta, datetime  

sec = 5

def remove_old_data(data):
    now = datetime.now() - timedelta(seconds=sec)
    #print(now)
    data = [(i,j) for i,j in data if j > now ]
    return data

def pipe_client():
    print("pipe client")
    quit = False
    data_cap = []
    while not quit:
        try:
            handle = win32file.CreateFile(
                r'\\.\pipe\Foo',
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
            res = win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
            if res == 0:
                print(f"SetNamedPipeHandleState return code: {res}")
            while True:
                
                resp = win32file.ReadFile(handle, 64*1024)
                data_cap.append((resp[1].decode(encoding='utf-8', errors='strict'), datetime.now()))
                if len(data_cap) == 1:
                    print("")
                data_cap = remove_old_data(data_cap)
                print([i for i,j in data_cap])
                #time.sleep(sec)
                #print(f"got message: {resp}")
        except pywintypes.error as e:
            if e.args[0] == 2:
                print("no pipe, trying again in a sec",end="\r")
                time.sleep(1)
            elif e.args[0] == 109:
                print("broken pipe, bye bye")
                while True:
                    data_cap = remove_old_data(data_cap) 
                    if data_cap:
                       print([i for i,j in data_cap])
                       time.sleep(1)
                    else:
                        break
                            
                quit = True


if __name__ == '__main__':
    pipe_client()

