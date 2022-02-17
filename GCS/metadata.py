#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, cv2
import time, re
import ffmpeg

import utils

import pandas as pd

from datetime import datetime, timedelta
from collections import deque, defaultdict

from mainclass import Main

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LNRMetadata(Main):
    """
    This class will extract metadata details from the LNR video.
    """
    
    
    def __init__(self, *args, **kwargs):    
        """
        Initializes all instance variable on the video
       
        """
        super().__init__(self, *args, **kwargs)
        
        self.filename = os.path.basename(self.filenamepath)
        self.filesize = os.stat(self.filenamepath).st_size // 1024
        self.filesize = '{0} KB'.format(self.filesize)
        self.filecreationtime = str(time.ctime(os.path.getmtime(self.filenamepath)))
        
        self.partialfile = False
        self.rawfileconverted = False
        
        
    def read_metadata(self, *args, **kwargs):
        """
        """
        
        # pull encoded metadata from file bytes
        
        with open(self.filenamepath, 'rb') as binary_file:
            binary_file.seek(-4, os.SEEK_END)
            val = binary_file.read(4).hex()
            if val.count('f') != 8:
                print('Invalid EOF in input video file. Got {0} instead of ffffffff. Using default values.'.format(val))
                self.starttimehexencoded = os.path.basename(self.filename).split(" ")[0]
                self.starttimeencoded = utils.get_utc_from_str(self.starttimehexencoded)
                self.heightencoded = 0
                self.widthencoded = 0
                self.duration = 0
                self.frame_rate = 10
                self.utc_endtime = datetime(9999, 12, 31)
                self.coded_frames = 999999
                self.cameraencoded = ""
                self.partialfile = True
                return
            
            binary_file.seek(32, os.SEEK_SET)
            self.starttimehexencoded = utils.get_value_from_hex(binary_file,8)
            self.starttimeencoded = utils.get_utc_from_str(self.starttimehexencoded)
            
            binary_file.seek(164, os.SEEK_SET)
            self.widthencoded = utils.get_value_from_hextoint(binary_file,2)           

            binary_file.seek(168, os.SEEK_SET)
            self.heightencoded = utils.get_value_from_hextoint(binary_file,2)           
            
            binary_file.seek(-1, os.SEEK_END)
            last_byte = binary_file.tell()
            
            binary_file.seek(-24, os.SEEK_END)
            lfadd = utils.get_value_from_hextoint(binary_file,4)       
            binary_file.seek(int(lfadd), os.SEEK_SET)
            lfsize = utils.get_value_from_hextoint(binary_file,4)       
            lfendadd = int(lfadd) + int(lfsize)
            binary_file.seek(int(lfendadd), os.SEEK_SET)
            findstr = deque(maxlen=8)
            cname = list()
            while '0000c800000000' not in ''.join(list(findstr)):
                tmp = binary_file.read(1)
                findstr.append(tmp.hex())
                cname.append(chr(int(tmp.hex(), 16)))            
            cameraname = ''.join(re.sub('[^A-Za-z0-9-/:]+', '',''.join(cname)))
            (_, delim, end) = cameraname.partition('US')
            (cameraname, _, _) = (delim + end).partition('/')
            self.cameraencoded = re.sub('([0-9]+)$', '', cameraname)            
            
            binary_file.seek(-17, os.SEEK_CUR)
            self.coded_frames = utils.get_value_from_hextoint(binary_file,4)     
            while self.coded_frames == 1:
                binary_file.seek(-24, os.SEEK_CUR)
                self.coded_frames = utils.get_value_from_hextoint(binary_file,4)

            frames = utils.get_value_from_hextoint(binary_file,8)
            
            cur_pos = binary_file.tell()
            coded_frames_new = (last_byte - cur_pos - 7) // 16
            self.frame_rate = round(self.coded_frames * 10000000 / frames)
            if self.frame_rate == 0:
                self.frame_rate = round(coded_frames_new * 10000000 / frames)
                self.coded_frames = coded_frames_new       

            self.utc_endtime = self.starttimeencoded + timedelta(seconds=round(self.coded_frames / self.frame_rate))
            self.duration = (self.utc_endtime - self.starttimeencoded).total_seconds()


    def check_file(self, *args, **kwargs):
        """
        """

        cap = cv2.VideoCapture(self.filenamepath)
        if cap.isOpened():
            cap.release()
            return
        
        print("Error opening input video file {0} using OpenCV".format(self.filenamepath))
        temp_file = self.filenamepath + '_temp.avi'

        out, _ = (
                    ffmpeg
                    .input(self.filenamepath, format=self.const_rawcodec, framerate=str(self.frame_rate))
                    .output(temp_file, crf=self.const_rawcrf, vcodec=self.const_outputvideocodec, video_size=(self.widthencoded,self.heightencoded))
                    .overwrite_output()
                    .run()
                 )
                
        
        os.rename(self.filenamepath, self.filenamepath + '_raw') 
        os.rename(temp_file, self.filenamepath) 
        self.rawfileconverted = True
            

    def save(self, *args, **kwargs):
        """
        Save information to csv file
        """
        
        temp_dict = defaultdict(dict)
        temp_dict['fileattributes'] =   {
                                        'filenamepath'          : self.filenamepath,
                                        'filename'              : self.filename,
                                        'filesize'              : self.filesize,
                                        'filecreationtime'      : self.filecreationtime,
                                        'heightencoded'         : self.heightencoded,
                                        'widthencoded'          : self.widthencoded,
                                        'framerateencoded'      : self.frame_rate,
                                        'totalframesencoded'    : self.coded_frames,
                                        'starttimehexencoded'   : self.starttimehexencoded,
                                        'starttimeencoded'      : self.starttimeencoded.strftime('%m/%d/%Y %I:%M:%S %p'),
                                        'endtime'               : self.utc_endtime.strftime('%m/%d/%Y %I:%M:%S %p'),
                                        'duration'              : self.duration,
                                        'cameranameencoded'     : self.cameraencoded,
                                        'filepartial'           : self.partialfile,                                                        
                                        'rawfileconverted'      : self.rawfileconverted,
                                       }
        
        self.file_df = pd.DataFrame.from_dict(temp_dict, 'index')

        super().save(self, *args, **kwargs)
        

            
