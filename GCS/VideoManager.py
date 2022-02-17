#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This is the manager module for video which stores video information. This will act as the meta data for the *.lnr files
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, time
import cv2, json
import VideoProcess, Utils
import sys

import numpy as np
import pandas as pd

from datetime import datetime
from collections import defaultdict
from Utils import Progress
from datetime import timedelta


class VideoFile:
    """
    This class will contain all the details about the video.
    """
    
    
    def __init__(self, *args, **kwargs):
        """
        Initializes all instance variable on the video
        """

        classlocation = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
        self.debug = args[5]
        self.settings = Utils.load_constants(self, classlocation, self.debug)
        
        self.process = args[1]
        self.processstarttime = datetime.utcnow()
        self.scanendtime = None
        self.heightencoded = 0
        self.widthencoded = 0
        self.duration = 0  # in secs
        self.framerateencoded = 0
        self.filenamepath = args[0]
        self.filename = os.path.basename(self.filenamepath)
        self.starttimeencoded = None
        self.starttimehexencoded = None
        self.endtime = None
        self.totalframesencoded = 0
        self.totalframes = 0
        self.cameraencoded = None
        self.filenamepathconverted = None
        # pull file attributes
        self.filesize = os.stat(self.filenamepath).st_size // 1024
        self.filecreationtime = str(time.ctime(os.path.getmtime(self.filenamepath)))
        

    def save_attributes(self, *args, **kwargs):
        """
        Save attributes to json file
        """
        
        self.processendtime = datetime.utcnow()
        
        temp_dict = defaultdict(dict)
        temp_dict[self.filename]['fileattributes'] =   {
                                                        'filenamepath'          : self.filenamepath,
                                                        'filenamepathconverted' : self.filenamepathconverted,
                                                        'filename'              : self.filename,
                                                        'filesize'              : '{0} KB'.format(self.filesize),
                                                        'filecreationtime'      : self.filecreationtime,
                                                        'filefourcc'            : self.filefourcc,
                                                        'filehex32changed'      : self.hex32changed,                                                        
                                                        'filepartial'           : self.partialfile,                                                        
                                                        'heightencoded'         : self.heightencoded,
                                                        'widthencoded'          : self.widthencoded,
                                                        'duration'              : self.duration,
                                                        'framerateencoded'      : self.framerateencoded,
                                                        'starttimehexencoded'   : self.starttimehexencoded,
                                                        'starttimeencoded'      : self.starttimeencoded.strftime('%m/%d/%Y %I:%M:%S %p'),
                                                        'endtime'               : self.endtime.strftime('%m/%d/%Y %I:%M:%S %p'),
                                                        'totalframesencoded'    : self.totalframesencoded,
                                                        'totalframes'           : self.totalframes,
                                                        'cameraencoded'         : self.cameraencoded,
                                                        
                                                       }

        temp_dict[self.filename][self.process] = {'starttime'         : self.processstarttime.strftime('%m/%d/%Y %I:%M:%S %p'),
                                                  'endtime'           : self.processendtime.strftime('%m/%d/%Y %I:%M:%S %p'),
                                                  'duration'          : round((self.processendtime - self.processstarttime).total_seconds(),2),
                                                  'runsettings'       : self.settings,
                                                 }

        totalprocessduration = 0
        dict_keys = [i for i in temp_dict[self.filename].keys() if i != 'fileattributes']
        for i in dict_keys:
            totalprocessduration += temp_dict[self.filename][i].get('duration',0)
        temp_dict[self.filename]['fileattributes'].update({'totalprocessduration': totalprocessduration})     
        
        with open(self.filenamepath + self.const_metadataext, 'w') as fp:
            json.dump(temp_dict, fp)

        self.df.to_parquet(self.filenamepath + self.const_metadatacompressionext, compression=self.const_metadatacompression)
        self.df.to_csv(self.filenamepath + '.csv',header=True, sep=',', index=False)

    
    def pull_metadata(self, *args, **kwargs):
        """        
        """
        
        # pull encoded metadata from bytes
        (
            self.heightencoded,
            self.widthencoded,
            self.duration,
            self.framerateencoded,
            self.starttimehexencoded,
            self.starttimeencoded,
            self.endtime,
            self.totalframesencoded,
            self.cameraencoded,
            self.hex32changed,
            self.partialfile,
            ) = VideoProcess.get_metadata(self.filenamepath)
        
        
        

    def process_frames(self, *args, **kwargs):
        """
        Reads the video and processes frame to extract metadata.
        It also performs background subtraction to remove motionless frames
        """
        
        
#        with open(self.filenamepath, 'r+b') as binary_file:
#            with open("videos\\1d4add6e1d9139c 173602.lnr", 'rb') as file:
#                update = file.read(200)
#                binary_file.write(update)
#            print("Timestamp removed from video file...")

#            self.hex32changed = True        
        
        activity = 'Scanning'

        cap = cv2.VideoCapture(self.filenamepath)
        if not cap.isOpened():
            raise Exception('Unable to open input video file {0}'.format(self.filenamepath))

        self.filefourcc = cap.get(cv2.CAP_PROP_FOURCC)
#        if not self.filefourcc:
#            raise Exception('Codec {0} is invalid'.format(self.filefourcc))

        if not self.heightencoded:
            cap1 = cv2.VideoCapture(self.filenamepath)
            (ret, frame) = cap.read()
            self.heightencoded = frame.shape[0]
            self.widthencoded = frame.shape[1]
            cap1.release()

        self.filenamepathconverted = self.filenamepath + self.const_videoext + self.const_videocontainer
        out = cv2.VideoWriter(self.filenamepathconverted ,cv2.VideoWriter_fourcc(*self.const_videocompression.upper()),self.framerateencoded, (self.widthencoded,self.heightencoded))    
        
        frame_count = 0
        
        if self.const_motioncapture:
            fgbg = cv2.createBackgroundSubtractorMOG2()
        
        temp_dict = {}
        
        with Progress(activity, self.totalframesencoded) as pbar:
            
            while True:
                (ret, frame) = cap.read()
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
        
        cap.release() 
        out.release()
        
        if not self.endtime:
            self.endtime = self.starttimeencoded + timedelta(seconds=round(self.totalframes / self.framerateencoded))
            self.duration = (self.endtime - self.starttimeencoded).total_seconds()
        
        if self.totalframesencoded == 999999:
            self.totalframesencoded = self.totalframes

        self.df = pd.DataFrame.from_dict(temp_dict, 'index')

        


        
        


