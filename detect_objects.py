import multiprocessing
#from imageai.Detection import ObjectDetection
import os
from datetime import timedelta, datetime
import numpy as np
#import argparse
from glob import iglob
import time
import cv2


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def apply_person_detection(filename, output_folder):

    model_path = './model'
    yolo_weight_path = os.path.join(model_path, 'yolov3.weights')
    yolo_config_path = os.path.join(model_path, 'yolov3.cfg')
    yolo_class_path = os.path.join(model_path, 'yolov3.txt')

    #detector = ObjectDetection()
    #detector.setModelTypeAsYOLOv3()
    #detector.setModelPath("./model/yolo.h5")
    #detector.loadModel(detection_speed="faster")
    #custom_objects = detector.CustomObjects()

    #name = os.path.basename(filename)
    #print(name)
    
    #output_file = os.path.join(output_folder,name)
    
    image = cv2.imread(filename)
    
    with open(yolo_class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]    
    #detections = detector.detectObjectsFromImage(input_image=frame, input_type='array', minimum_percentage_probability=50)
    #detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=frame, input_type='array', minimum_percentage_probability=50)
    #print(detections)

# =============================================================================
#     for index, eachObject in enumerate(detections):
#         #print(eachObject)
#         name = eachObject['name']
#         #percentage_probability = eachObject['percentage_probability']
#         x1, y1, x2, y2 = eachObject['box_points']
#         file_n = os.path.join(output_folder,'{}_{}_{}.jpg'.format(os.path.basename(filename),name,str(index)))
#         #cv2.imwrite(file_n + 'x1y1x2y2.jpg',frame[x1: y1, x2: y2])
#         #cv2.imwrite(file_n + 'x2y2x1y1.jpg',frame[x2: y2, x1: y1])
#         #cv2.imwrite(file_n + 'x1y2x2y1.jpg',frame[x1: y2, x2: y1])
#         #cv2.imwrite(file_n + 'x2y1x1y2.jpg',frame[x2: y1, x1: y2])
# 
#         cv2.imwrite(file_n + 'y1x1y2x2.jpg',frame[y1: x1, y2: x2])
#         #cv2.imwrite(file_n + 'y2x2y1x1.jpg',frame[y2: x2, y1: x1])
#         #cv2.imwrite(file_n + 'y1x2y2x1.jpg',frame[y1: x2, y2: x1])
#         #cv2.imwrite(file_n + 'y2x1y1x2.jpg',frame[y2: x1, y1: x2])
# =============================================================================
       
    width = image.shape[1]
    height = image.shape[0]
    scale = 0.00392
    net = cv2.dnn.readNet(yolo_weight_path,yolo_config_path)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    
    net.setInput(blob)
    
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.3
    nms_boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: # and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    if len(indices) > 0 :
        for cnt, i in enumerate(indices):
            i = i[0]
            box = list(map(int,boxes[i]))
            x = max(box[0],0)
            y = max(box[1],0)
            w = box[2]
            h = box[3]
            nms_boxes.append([x, y, w, h])
            file_n = os.path.join(output_folder,'{}_{}_{}.jpg'.format(os.path.basename(filename),str(classes[class_ids[i]]),str(cnt)))
            cv2.imwrite(file_n,image[y: y + h, x: x + w])
    
    return nms_boxes        
    



if __name__ == '__main__':
    start = time.monotonic()

#    ap = argparse.ArgumentParser()
#    ap.add_argument("-i", "--input", required=True,
#            help="Path to input videos folder")
#    ap.add_argument("-o", "--output", required=True,
#            help="Path to output videos folder")
#    args = vars(ap.parse_args())
    
#    output_folder =  args["output"]
    output_folder = './output'
    input_folder = './input'
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    tasks = list()
    
    for filename in iglob(os.path.join(input_folder,'*'), recursive=False):
        tasks.append(multiprocessing.Process(target=apply_person_detection, args=(os.path.abspath(filename),output_folder )))

    [result.start() for result in tasks]
    [result.join() for result in tasks]


    d = datetime(1, 1, 1) + timedelta(seconds=time.monotonic() - start)
    print('\nElapsed Time : {0}'.format(timedelta(seconds=time.monotonic() - start)))