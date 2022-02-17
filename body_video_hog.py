import cv2
from imutils.object_detection import non_max_suppression
#import time
import imutils
import numpy as np
import tensorflow as tf
import os
#from detect_object import detect_objects
CWD_PATH = os.getcwd()
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_MODEL = os.path.join(CWD_PATH, 'HC', 'frozen_inference_graph.pb')

# Create a VideoCapture object
cap = cv2.VideoCapture("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\op_533396.mp4")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#face_cascade =  cv2.CascadeClassifier("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\\mtcnn\\HC\\haarcascade_upperbody.xml")
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter("outpy.avi",cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width,frame_height))
#out = cv2.VideoWriter('D:\\Arasan\\Misc\\GitHub\\FaceNetM\\outpy.mp4',cv2.VideoWriter_fourcc('M','P','V','4'), 20.0, (640,480))
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
c = 0

while(True):
    #time.sleep(20)
    ret, frame = cap.read()
    if ret == True: 
        try:
            #frame = imutils.resize(frame, width=min(400, frame.shape[1]))
            (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
            orig = frame.copy()
            for x, y, w, h in rects:
                frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            #pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            #for (xA, yA, xB, yB) in pick:
            #    cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)
            #out.write(frame)
            c += 1
            print('{0} --> {1}'.format(c,length))
        except Exception as e:
            print(e)
        
    
        # Display the resulting frame    
        cv2.imshow('frame',frame)
        #cv2.imshow('orig',orig)
 
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
  # Break the loop
    else:
        break 
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 