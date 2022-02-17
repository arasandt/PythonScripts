#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains all functions which would manipulate on a single image
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2

from Utils import Progress
from PIL import Image


def get_images(*args, **kwargs): 
    """
    """

    filenamepath = args[0]
    required_frames = args[1]
    outwrite    = kwargs.get('outwrite', False)
    applyresult = kwargs.get('applyresult', [])
    fmt         = kwargs.get('fmt', False)

    cap = cv2.VideoCapture(filenamepath)
    frame_count = 1    
    
    activity = 'Writing Image(s)' if outwrite else 'Pulling Image(s)'
    
    with Progress(activity, len(required_frames)) as pbar:
        
        for tup in range(len(required_frames)):
            frame, group = required_frames[tup]
        
            while frame_count <= frame:
                cap.grab()
                frame_count += 1
            ret, image = cap.retrieve()
       
            if outwrite:
                if fmt:
                    im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    im.save(outwrite, fmt) 
                else:
                    if applyresult:
                        result = [] if applyresult[tup] == 0 else eval(applyresult[tup])
                    else:
                        result = []
                    image = apply_model_result(frame, group, image, result)
                    outwrite.write(image)
            else:
                required_frames[tup] = (frame, group, image)
        
            pbar.update(tup + 1)
    
    if not outwrite:
        return required_frames


def apply_model_result(*args, **kwargs): 
    """
    """
    
    frame, group, image, result = args
    showmessage = kwargs.get('showmessage', False)
    
    for coor, confi in result:
        image = cv2.rectangle(image, (coor[0], coor[1]), (coor[0] + coor[2], coor[1] + coor[3]), (255,255,0), 1)
        cv2.putText(image, str(confi), (coor[0], coor[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
        
        if showmessage:
            print('Frame #{0} => Confidence(%): {1}  Co-ordinates(x,y,w,h): {2}'.format(frame,int(confi*100),coor))
    
    cv2.putText(image, str(int(frame)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), thickness=1, lineType=cv2.LINE_8)
    return image
