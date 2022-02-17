#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
import cv2
import pandas as pd
import numpy as np

from datetime import timedelta

from core.mainclass import Main
from core import utils
from core.utils import Progress, cv2Pseudo


class ScanVideoFile(Main):
    """
    """
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        super().__init__(self, *args, **kwargs)
        self.file_df = pd.read_csv(self.filenamepath + self.const_fileext)
        self.filepartial        = self.file_df.iloc[0]['filepartial']
        self.framerateencoded   = self.file_df.iloc[0]['framerateencoded']
        self.heightencoded      = self.file_df.iloc[0]['heightencoded']
        self.widthencoded       = self.file_df.iloc[0]['widthencoded']
        self.totalframesencoded = self.file_df.iloc[0]['totalframesencoded']
        self.starttimehexencoded = self.file_df.iloc[0]['starttimehexencoded']
        self.starttimeencoded_new   = utils.get_utc_from_str(self.starttimehexencoded)

        
    def process_frames(self, *args, **kwargs):
        """
        Reads the video and processes frame to extract metadata.
        It also performs background subtraction to remove motionless frames
        """
        
        activity = 'Scanning'

        cv2Obj = cv2Pseudo(self.filenamepath, suppressffmpeg=self.const_suppressffmpeg)
        cv2Obj.VideoCapture()
        
        self.filefourcc = cv2Obj.get(cv2.CAP_PROP_FOURCC)
        
        self.filenamepathconverted = self.filenamepath + self.const_videoext + self.const_videocontainer
        out = cv2.VideoWriter(self.filenamepathconverted ,cv2.VideoWriter_fourcc(*self.const_videocompression.upper()),self.framerateencoded, (self.widthencoded,self.heightencoded))    

        frame_count = 0
        
        if self.const_motioncapture:
            fgbg = cv2.createBackgroundSubtractorMOG2()
        
        temp_dict = {}
        
        with Progress(activity, self.totalframesencoded) as pbar:
            
            while True:
                (ret, frame) = cv2Obj.read()
                if not ret:
                    break

                orig_frame = frame.copy()
                
                if self.const_motioncapture:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.GaussianBlur(frame, self.const_motionblur, 0)
                    fgmask = fgbg.apply(frame, learningRate=self.const_motionlearningrate)
                
                if frame_count == 0:
                    fgmean = 0 if self.const_motioncapture else 1
                else:
                    fgmean = np.mean(fgmask) * 100 / 255 if self.const_motioncapture else 1
                
                temp_dict[frame_count] = {'framenumber': frame_count + 1, 
                                          'fgmean': fgmean} 
    
                out.write(orig_frame)
    
                frame_count += 1

                pbar.update(frame_count)
            

        self.totalframes = frame_count
        
        cv2Obj.release() 
        out.release()
        
        if self.filepartial:
            self.endtime = self.starttimeencoded_new + timedelta(seconds=round(self.totalframes / self.framerateencoded))
            self.duration = (self.endtime - self.starttimeencoded_new).total_seconds()
            self.totalframesencoded = self.totalframes
            self.file_df['endtime'] = self.endtime
            self.file_df['duration'] = self.duration
            self.file_df['totalframesencoded'] = self.totalframesencoded

        self.frame_df = pd.DataFrame.from_dict(temp_dict, 'index')
            
    
    def save(self, *args, **kwargs):
        """
        """
        
        self.file_df['totalframes'] = self.totalframes
        self.file_df['filefourcc'] = self.filefourcc
        self.file_df['filenamepathconverted'] = self.filenamepathconverted
        
        super().save(self, *args, **kwargs)
        