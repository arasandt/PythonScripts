#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import cv2

import pandas as pd

from mainclass import Main

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ValidateOutput(Main):
    """
    """
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        super().__init__(self, *args, **kwargs)
        self.file_df         = pd.read_csv(self.filenamepath + self.const_fileext)
        self.frame_df        = pd.read_csv(self.filenamepath + self.const_frameext)
        
    
    def verify_output(self, *args, **kwargs):
        """
        """
        
        df = self.frame_df.loc[(self.frame_df['writeflag'] != 0.0)][['framenumber','framegroup']]
        maxnnewframe = self.frame_df['newframenumber'].max()
        
        if len(df):
            opname = self.file_df.iloc[0]['outputfilenamepath']        
            cap = cv2.VideoCapture(opname)
            attributesop = {
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    		                 }
            cap.release()
            
            attributesin = {
                            "height": self.file_df.iloc[0]['heightencoded'],
                            "width": self.file_df.iloc[0]['widthencoded'],
                            "fps": self.file_df.iloc[0]['framerateencoded'],
                            "frames": maxnnewframe,
    		                 }        
            
            
            assert attributesin == attributesop, 'Validation failure between input and output\nInput  {0}\nOutput {1}'.format(attributesin, attributesop)
            

    def save(self, *args, **kwargs):
        """
        """
        
        super().save(self, *args, **kwargs)                   