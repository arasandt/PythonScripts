#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2, os
import Utils, sys

import numpy as np


class ModelObject:
    """
    """


    def assign_modelfiles(self):
        """
        """
        last_folder = os.path.basename(os.path.normpath(self.modelfolder))
        
        self.model_weight_path        = os.path.join(self.modelfolder, last_folder + '.weights')
        self.model_config_path        = os.path.join(self.modelfolder, last_folder + '.cfg')
        self.model_class_path         = os.path.join(self.modelfolder, last_folder + '.txt')
    

    def __init__(self, *args, **kwargs):
        """
        """
        self.debug = kwargs.get('debug', False)
        classlocation = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
        self.settings = Utils.load_constants(self, classlocation, self.debug)

        self.filenamepath = args[0]
        self.modelfolder  = args[1] 
        self.filename = os.path.basename(self.filenamepath)
        self.image = kwargs.get('image', None)
        self.framenumber = kwargs.get('framenumber', None)
        self.assign_modelfiles()
        
        with open(self.model_class_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]            
        
        self.net = cv2.dnn.readNet(self.model_weight_path, self.model_config_path)

    
    def get_output_layers(self, net):
        """
        """
        
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers


    def setup_model(self, *args, **kwargs):
        """
        """

        blob = cv2.dnn.blobFromImage(self.image, self.const_nnscale, self.const_nnspatialsize, self.const_nnmeansubtraction, self.const_nnswapRB, crop=self.const_nncrop)
        self.net.setInput(blob)
        self.nnoutput = self.net.forward(self.get_output_layers(self.net))            


    def detect(self, *args, **kwargs):
        """
        """
        
        self.setup_model()
        
        width = self.image.shape[1]
        height = self.image.shape[0]         
        confidences = []
        boxes = []        

        for out in self.nnoutput:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.const_modelminconfidence and class_id == self.const_modelpersonclassid:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        if len(boxes):
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.const_modelnmsconfidence, self.const_modelnmsthreshold)
        else:
            indices = []
        
        if len(indices):
            # get bounding box, confidence and area % into result
            result = [(j[0], round(j[1],2)) for i, j in enumerate(list(zip(boxes,confidences))) if i in np.reshape(indices,(1,len(indices)))[0]]
            
            # result should be more than x% of original
            #result = [(i,j,k) for i, j, k in result if (w * h) >= (width * height * self.const_modeldetectionareathreshold)]
            return result
        else:
            return -1
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        