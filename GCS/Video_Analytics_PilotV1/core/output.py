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

import os, cv2
import ffmpeg
import pandas as pd

from pathlib import Path

from core.mainclass import Main
from core import utils


class OutputVideoFile(Main):
    """
    """
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        super().__init__(self, *args, **kwargs)
        self.file_df         = pd.read_csv(self.filenamepath + self.const_fileext)
        self.frame_df        = pd.read_csv(self.filenamepath + self.const_frameext)
        
        self.file_df = self.file_df.fillna("")
        self.filenamepathconverted = self.file_df.iloc[0]['filenamepathconverted']
        if not self.filenamepathconverted or not Path(self.filenamepathconverted).exists():
            raise Exception('Converted input video file {0} not found'.format(self.filenamepathconverted))        
        
        self.const_videoextdef   = '.{0}.'.format(self.const_videocompressiondef)        
        self.const_ffmpegextdef  = '.{0}.'.format(self.const_ffmpegcompressiondef)        
       
    
    def write_model_frames(self, *args, **kwargs):
        """
        """
        
        fps = self.file_df.iloc[0]['framerateencoded']
        width = self.file_df.iloc[0]['widthencoded']
        height = self.file_df.iloc[0]['heightencoded']
        
        df = self.frame_df.loc[(self.frame_df['writeflag'] != 0.0)][['framenumber','framegroup']]
        self.outputfileframes = len(df)
        
        self.outputfilenamepath = ""
                
        if len(df):
            inputfilename = self.filenamepath + self.const_videoextdef + self.const_videocontainerdef
            out = cv2.VideoWriter(inputfilename, cv2.VideoWriter_fourcc(*self.const_videocompressiondef.upper()),fps, (width,height))    
            
            print('Selecting {0} image(s)'.format(len(df)), flush=True)
            
            utils.get_images(self.filenamepathconverted, 
                             list(zip(df['framenumber'],df['framegroup'])), 
                             outwrite=out)
            
            out.release()
            
            self.outputfilenamepath = self.filenamepath + self.const_ffmpegextdef + self.const_ffmegcontainerdef
            
            out, _ = (
                        ffmpeg
                        .input(inputfilename)
                        .output(self.outputfilenamepath, crf=self.const_ffmpegcrfdef, vcodec=self.const_ffmpegcompressiondef)
                        .overwrite_output()
                        .run()
                     )
            
            # remove the temp video file    
            os.remove(inputfilename)
        else:
            print('No output video file generated', flush=True)
        

    def save(self, *args, **kwargs):
        """
        """
        
        self.file_df['outputfilenamepath'] = self.outputfilenamepath
        self.file_df['outputfileframes']   = self.outputfileframes
        
        super().save(self, *args, **kwargs)          