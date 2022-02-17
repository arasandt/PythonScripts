import glob
import cv2
import os
from PIL import Image
from io import BytesIO
import numpy as np
import math
from mtcnn.mtcnn import MTCNN
detector = MTCNN()



def equalize(f):
    h = np.histogram(f, bins=np.arange(257))[0]
    H = np.cumsum(h) / float(np.sum(h))
    e = np.floor(H[f.flatten().astype('int')]*255.)
    return e.reshape(f.shape)

def pullfaces(frame,c):
    result_all = detector.detect_faces(frame)
    
    if result_all:
        for i,result in enumerate(result_all):
            print(result)
            if result['confidence'] >= 0.0:
            #if result['confidence'] <= 0.75:
            #    cv2.imwrite(fil + str(c) + str(i) + str(result['confidence']) + ".lessconfib.png", frame)
                x = result['box'][0]
                y = result['box'][1]
                w = result['box'][2]
                h = result['box'][3]
                cv2.rectangle(frame,(x, y),(x + w, y + h),(0,155,255), 2)
            #if result['confidence'] <= 0.75:
            #    cv2.imwrite(fil + str(c) + str(i) + str(result['confidence']) + ".lessconfia.png", frame)

    else:
        pass #print('No faced found')
    return frame


fpath = 'D:\\Arasan\\Misc\\GitHub\\VideoCapture\\MotionDetection\\Met1MDFEntrance.mp4.outpy.avi'
#dpath = 'D:\\Arasan\\Misc\\GitHub\\VideoCapture\\mtcnn\\'
for fil in glob.glob(fpath):
    cap = cv2.VideoCapture(fil)
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
        continue
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    c = 0
    out = cv2.VideoWriter(fil + ".avi",cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width,frame_height))
    while(True):
        ret, frame = cap.read()
        #if ret == True and c == 100: 
        if ret == True:
            try:
                #cv2.imwrite(fil + ".withface.png", frame)
                out.write(pullfaces(frame, c))
                cv2.imshow('frame',frame)
            except Exception as e:
                print(e)
        else:
            break 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        c += 1
        print('{0} --> {1}'.format(c,length))
    
    cap.release()
    out.release()
    cv2.destroyAllWindows() 