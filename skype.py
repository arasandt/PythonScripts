#from pywinauto.findwindows import find_window
#from pywinauto.win32functions import SetForegroundWindow
#SetForegroundWindow(find_window(title='Skype for Business'))

import win32gui
import re, os
from PIL import ImageGrab
from datetime import datetime
import time
import win32con


class WindowMgr:
    """Encapsulates some calls to the winapi for window management"""

    def __init__ (self):
        """Constructor"""
        self._handle = None
        self._fore_handle = None

    def find_window(self, class_name, window_name=None):
        """find a window by its class_name"""
        self._handle = win32gui.FindWindow(class_name, window_name)

    def _window_enum_callback(self, hwnd, wildcard):
        """Pass to win32gui.EnumWindows() to check all the opened windows"""
        if re.match(wildcard, str(win32gui.GetWindowText(hwnd))) is not None:
            self._handle = hwnd

    def find_window_wildcard(self, wildcard):
        """find a window whose title matches the wildcard regex"""
        self._handle = None
        win32gui.EnumWindows(self._window_enum_callback, wildcard)

    def set_foreground(self):
        """put the window in the foreground"""
        win32gui.SetForegroundWindow(self._handle)

    def get_current_foreground(self):
        """get the window in the foreground"""
        self._fore_handle = win32gui.GetForegroundWindow()

    def set_current_foreground(self):
        """get the window in the foreground"""
        if self._fore_handle is not None:
            win32gui.SetForegroundWindow(self._fore_handle)

w = WindowMgr()


folder = 'skypelog'
#for i in range(1,3):
while True:
    w.get_current_foreground()    
    w.find_window_wildcard(".*Skype for Business.*")
    try:
    	w.set_foreground()
    except:
    	time.sleep(300)
    	continue
    win32gui.ShowWindow(w._handle, win32con.SW_RESTORE)
    times = datetime.now()
    
    bbox = win32gui.GetWindowRect(w._handle)
    img = ImageGrab.grab(bbox)
    w.set_current_foreground()
    #img.save('out_' + str(times) + '.bmp')
    filename = os.path.join(folder, 'out_' + times.strftime('%m-%d-%Y-%I-%M-%S-%p-%Z') + '.jpg')
    print(filename)
    img.save(filename)
    time.sleep(600)

#print(type(img))
#img.show()



#toplist, winlist = [], []
#def enum_cb(hwnd, results):
#    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
#win32gui.EnumWindows(enum_cb, toplist)
#
#firefox = [(hwnd, title) for hwnd, title in winlist if 'Skype for Business' in title.lower()]
## just grab the hwnd for first window matching firefox
#firefox = firefox[0]
#hwnd = firefox[0]
#
#win32gui.SetForegroundWindow(hwnd)
#bbox = win32gui.GetWindowRect(hwnd)
#img = ImageGrab.grab(bbox)
#img.show()