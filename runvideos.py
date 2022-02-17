import numpy as np
import cv2
import time

#names = ['abc1 body.avi', 'abc1 faces.avi'];
names=["D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Juice Bar 10fps.lnr.output.mp4","D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Juice Bar 20fps.lnr.output.mp4",]
window_titles = ['10fps', '20fps']


cap = [cv2.VideoCapture(i) for i in names]

frames = [None] * len(names);
gray = [None] * len(names);
ret = [None] * len(names);

while True:

    for i,c in enumerate(cap):
        if c is not None:
            ret[i], frames[i] = c.read();


    for i,f in enumerate(frames):
        if ret[i] is True:
            gray[i] = f #cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            cv2.imshow(window_titles[i], gray[i]);
            #time.sleep(0.03)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


for c in cap:
    if c is not None:
        c.release();

cv2.destroyAllWindows()