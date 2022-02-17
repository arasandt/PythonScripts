import glob
import cv2
import os
from PIL import Image
from io import BytesIO
import numpy as np
import math
import time

from detect_object import detect_objects
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

CWD_PATH = os.getcwd()
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_MODEL = os.path.join(CWD_PATH, 'HC', 'frozen_inference_graph.pb')


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

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)


def isInRect(pt1,pt2,w1,h1,pt3,pt4,w2,h2):
    return (pt1 <= pt3) and (pt2 <= pt4) and (pt1 + w1) >= (pt3 + w2) and (pt2 + h1) >= (pt4 + h2)
    

fpath = 'D:\\Arasan\\Misc\\GitHub\\VideoCapture\\mtcnn\\*1.mp4'
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
                faceframe = np.copy(frame)
                bodyframe = np.copy(frame)
                #faceframe = pullfaces(faceframe, c)
                result = detector.detect_faces(faceframe)
                print(result)
                for j,i in enumerate(result):
                    x = i['box'][0]
                    y = i['box'][1]
                    w = i['box'][2]
                    h = i['box'][3]
                    if (w * h) >= 0: #1200: # not sure what to do with overlapping boxes
# =============================================================================
#                         if i['confidence'] < 0.85:
#                             body_context = detect_objects(bodyframe, sess, detection_graph)
#                             body_context['width'] = bodyframe.shape[1]
#                             body_context['height'] = bodyframe.shape[0]
#                             for point, name, color in zip(body_context['rect_points'], body_context['class_names'], body_context['class_colors']):
#                                 if 'person' in name[0]:
#                                     if isInRect(point['xmin'],point['ymin'],body_context['width'],body_context['height'],x,y,w,h):
#                                         cv2.rectangle(faceframe,(x, y),(x + w, y + h),(0,155,255), 2)
#                                         cv2.imwrite('./' + str(c) + '.' + str(i['confidence']) + ".concentric.png", faceframe)
#                                         cv2.rectangle(faceframe, (int(point['xmin'] * body_context['width']), int(point['ymin'] * body_context['height'])),
#                                                       (int(point['xmax'] * body_context['width']), int(point['ymax'] * body_context['height'])), color, 3)
#                         else:
# =============================================================================
                        if i['confidence'] > 0.85:
                            cv2.rectangle(faceframe,(x, y),(x + w, y + h),(0,155,255), 2)
# =============================================================================
#                         if j == 2:
#                             time.sleep(20)
#                             print("more than 2 face detected")
#                             cv2.imwrite('./' + str(c) + '.' + str(i['confidence']) + ".moreface.png", faceframe)
# =============================================================================
                                        
                                    #cv2.rectangle(faceframe, (int(point['xmin'] * body_context['width']), int(point['ymin'] * body_context['height'])),
                                    #              (int(point['xmax'] * body_context['width']), int(point['ymax'] * body_context['height'])), color, 3)

                #out.write()
                #cv2.imshow('Original',frame)
                #cv2.imshow('Body',bodyframe)
                cv2.imshow('Face',faceframe)
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