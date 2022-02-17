
# coding: utf-8

# In[535]:


import numpy as np
import sys
import cv2, os
from collections import deque
path = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\" + str(sys.argv[1]) # + ".mp4"
#path = os.path.abspath(str(sys.argv[1]))
cap = cv2.VideoCapture(path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_rate = 10
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_rate)
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter(os.path.basename(path) + ".outpy.avi",cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width * 2,frame_height))
out = cv2.VideoWriter(os.path.basename(path) + ".outpy.avi",cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width,frame_height))
#out1 = cv2.VideoWriter("outpy1.avi",cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width * 2,frame_height))


# In[536]:


#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False,varThreshold=32)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbg.setVarThreshold(60)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=35)#, nmixtures=5, backgroundRatio=0.7) #,noiseSigma=0)
#fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=5,backgroundRatio=0.1,noiseSigma=0)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#fgbg1 = cv2.bgsegm.createBackgroundSubtractorGMG()


# In[ ]:

c = 0
#motion = 0
first30 = deque(maxlen=frame_rate)
#next30 = list()
while(1):
    ret, frame = cap.read()
    if ret == False: break
    orig_frame = frame.copy()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    frame = cv2.GaussianBlur(frame, (21, 21), 0) # will depend on the resolution of the image
    #fgmask = fgbg.apply(frame)
    fgmask = fgbg.apply(frame, learningRate=0.001)
    position = int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
    print('{0} --> {1} --> {2}'.format(c,length,position))
    #print(position)
    #fgmask1 = fgbg1.apply(frame)
    #fgmask1 = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('frame', fgmask)
    #cv2.imshow('ori_frame', orig_frame)
    fgmean = np.mean(fgmask) * 100 / 255
    #if fgmean != 0:
    #    print(c,length,fgmean)
    if fgmean > 0.01:
        #print('{0} --> {1}'.format(c,length))
        #motion = 1
        #print(c,length,fgmean)
# =============================================================================
#         while True:
#             try:
#                 x, y = first30.popleft()
#                 #vis = np.concatenate((cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), y), axis=1)
#                 #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
#                 vis = np.concatenate((x,cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)), axis=1)
#                 #print('{0} --> {1} --> {2}'.format(c,length,len(first30)))
#                 #cv2.imshow('frame', vis)
#                 vis = x
#                 out.write(vis)
#             except:
#                 break
#         #vis = np.concatenate((cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY) , fgmask), axis=1)
# =============================================================================
        #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        vis = np.concatenate((orig_frame,cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)), axis=1)
        vis = orig_frame
        #cv2.imshow('frame', vis)
        out.write(vis)
    else:
        #if next30:
        #    for x,y in next30:
        #        vis = np.concatenate((x,cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)), axis=1)
        #        out.write(vis)
        pass
        #first30.append((orig_frame,fgmask))
        #if motion == 1 and len(first30) == frame_rate:
        #    next30 = list(first30)
        #    motion = 0
            
            
    c += 1
    #cv2.imshow('orig_frame', orig_frame)
    
    #print('{0} --> {1}'.format(c,length)) #, end='\r')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()

