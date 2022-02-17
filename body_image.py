import cv2
import tensorflow as tf
import os
from detect_object import detect_objects
CWD_PATH = os.getcwd()

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_MODEL = os.path.join(CWD_PATH, 'HC', 'frozen_inference_graph.pb')
#PATH_TO_VIDEO = os.path.join(CWD_PATH, 'input.mp4')

#face_cascade =  cv2.CascadeClassifier("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\\mtcnn\\HC\\haarcascade_fullbody.xml")
face_cascade =  cv2.CascadeClassifier("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\\mtcnn\\HC\\haarcascade_upperbody.xml")
#face_cascade =  cv2.CascadeClassifier("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\\mtcnn\\HC\\haarcascade_frontalface_alt.xml")
frame=cv2.imread("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\mtcnn\\abc.jpg")
gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

context = detect_objects(frame, sess, detection_graph)
print(context)
context['width'] = frame.shape[1]
context['height'] = frame.shape[0]


font = cv2.FONT_HERSHEY_SIMPLEX
for point, name, color in zip(context['rect_points'], context['class_names'], context['class_colors']):
    cv2.rectangle(frame, (int(point['xmin'] * context['width']), int(point['ymin'] * context['height'])),
                  (int(point['xmax'] * context['width']), int(point['ymax'] * context['height'])), color, 3)
    #cv2.rectangle(frame, (int(point['xmin'] * context['width']), int(point['ymin'] * context['height'])),
    #              (int(point['xmin'] * context['width']) + len(name[0]) * 6,
    #               int(point['ymin'] * context['height']) - 10), color, -1, cv2.LINE_AA)
    #cv2.putText(frame, name[0], (int(point['xmin'] * context['width']), int(point['ymin'] * context['height'])), font,
    #            0.3, (0, 0, 0), 1)



# =============================================================================
# faces=face_cascade.detectMultiScale(gray_img,
#                                     scaleFactor=1.1,
#                                     minNeighbors=5)
# for x, y, w, h in faces:
#     img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
#
# print(type(faces))
# print(faces)
# =============================================================================

#resized=cv2.resize(frame,(frame.shape[1]//3,frame.shape[0]//3))

cv2.imshow('grey',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()