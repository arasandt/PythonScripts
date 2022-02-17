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
import time
import pandas as pd

from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

from core.mainclass import Main
from core import utils
from core.utils import range_group, cv2Pseudo


class LNRMetadata(Main):
    """
    This class will extract metadata details from the LNR video.
    """
    
    
    def __init__(self, *args, **kwargs):    
        """
        Initializes all instance variable on the video
       
        """
        super().__init__(self, *args, **kwargs)
        
        self.p = Path(self.filenamepath)
        self.filename = self.p.name
        self.filesize = self.p.stat().st_size // 1024
        self.filesize = '{0} KB'.format(self.filesize)
        self.filecreationtime = str(time.ctime(self.p.stat().st_mtime))
        
        self.partialfile = False
        
        
    def read_metadata(self, *args, **kwargs):
        """
        """
        
        # pull encoded metadata from file bytes
        
        with open(self.filenamepath, 'rb') as binary_file:
            binary_file.seek(-4, os.SEEK_END)
            val = binary_file.read(4).hex()
            if val.count('f') != 8:
                print('Invalid EOF in input video file. Got {0} instead of ffffffff. Using default metadata values.'.format(val), flush=True)
                self.starttimehexencoded = self.filename.split(" ")[0]
                self.starttimeencoded = utils.get_utc_from_str(self.starttimehexencoded)
                self.heightencoded = 0
                self.widthencoded = 0
                self.duration = 0
                self.frame_rate = self.dbfps
                self.utc_endtime = datetime(9999, 12, 31)
                self.coded_frames = 99999
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
            
            self.frame_rate  = self.dbfps
            self.coded_frames = self.pull_bytes(binary_file)

            self.cameraencoded = ""
            self.utc_endtime = self.starttimeencoded + timedelta(seconds=round(self.coded_frames / self.frame_rate))
            self.duration = (self.utc_endtime - self.starttimeencoded).total_seconds()


    def pull_bytes(self, binary_file):
        """
        """
        
        temp_dict = defaultdict(dict)
        position = 200
        binary_file.seek(position, os.SEEK_SET)
        cnt = 0
        file_size = self.p.stat().st_size
        while position <= file_size:
            val = utils.get_value_from_hextoint(binary_file, 3)
            if val == 0: break;
            binary_file.seek(val - 3, os.SEEK_CUR)
            
            temp_dict[cnt] = {
                            'curposition'   : position,
                            'size'          : val,
                            #'keyframehexid' : keyframehex,
                            'nextposition'  : position + val,
                           }
            
            position = position + val
            cnt += 1
        
        temp_df = pd.DataFrame.from_dict(temp_dict, 'index')
        firstfpstrows = temp_df.head(self.frame_rate)
        self.maxsizeindex = firstfpstrows['size'].idxmax(axis = 0)
        self.maxposition = firstfpstrows.iloc[self.maxsizeindex]['curposition']
        #temp_df = temp_df.loc[maxsizeindex:,:]
        temp_df.to_csv(self.filenamepath + self.const_framebytesext, header=True, sep=',', index=False)                
        return len(temp_df)
    
    def check_file(self, *args, **kwargs):
        """
        """
        
        cv2Obj = cv2Pseudo(self.filenamepath, suppressffmpeg=self.const_suppressffmpeg)
        cv2Obj.VideoCapture()
        if cv2Obj.isOpened():
            print("Opening input video file {0} using OpenCV successful".format(self.filenamepath), flush=True)
            (ret, frame) = cv2Obj.read()
            self.heightencoded = frame.shape[0]
            self.widthencoded = frame.shape[1]
            cv2Obj.release()
            return

        raise Exception ("Error opening input video file {0} using OpenCV".format(self.filenamepath))
        

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
                                       }
        
        self.file_df = pd.DataFrame.from_dict(temp_dict, 'index')

        super().save(self, *args, **kwargs)
        
           
class PrepareData(Main):
    """
    """
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        super().__init__(self, *args, **kwargs)
        self.file_df = pd.read_csv(self.filenamepath + self.const_fileext)
        self.frame_df = pd.read_csv(self.filenamepath + self.const_frameext)
        
   
    def prep_process(self, *args, **kwargs):
        """
        """
        
        #get fps & total frames
        fps                = self.file_df.iloc[0]['framerateencoded']
        totalframes        = self.file_df.iloc[0]['totalframesencoded']
        
        # clean columns
        self.frame_df.drop(['fgmeangroupsum','fgmeangrouprank','framegroup'], axis=1, inplace=True, errors='ignore')

        #check if background subtraction was applied
        motionapplied = len(self.frame_df.loc[(self.frame_df['fgmean'] != 1)])

        # group frame into 1 sec batches
        self.frame_df['framegroup'] = 1 + (self.frame_df['framenumber'] - 1) // fps
        
        # apply fgmean sum to all batch records and get all motion batches
        df = self.frame_df.groupby(['framegroup'], as_index=False)['fgmean'].mean()
        df.rename(columns = {'fgmean':'fgmeangroupsum'}, inplace = True) 
        
        if motionapplied:
            dfm = df[df['fgmeangroupsum'] > self.const_fgmeanthreshold].copy()
            frameswithmotion = len(dfm) * fps
        else:
            frameswithmotion = len(df) * fps
        
        if self.const_applymotioncapture:
            df = df[df['fgmeangroupsum'] > self.const_fgmeanthreshold]

        # merge metadata with motion data
        self.frame_df = pd.merge(self.frame_df, df, on='framegroup',how='left').fillna(0)

        # rank the fgmean values
        self.frame_df['temprank'] = self.frame_df.groupby(['framegroup'], as_index=False)['fgmean'].rank("min", ascending=False)
        self.frame_df['fgmeangrouprank'] = self.frame_df.sort_values(['temprank']).groupby(['framegroup']).cumcount() + 1
        self.frame_df.drop(['temprank'], axis=1, inplace=True, errors='ignore')
        
        motionpercentage = round((frameswithmotion * 100.0 / totalframes),2)
        self.motionpercentage = min(motionpercentage, 100.00)
        

    def save(self, *args, **kwargs):
        """
        """
        
        self.file_df['motionpercentage'] = self.motionpercentage
        
        super().save(self, *args, **kwargs)        
        
        
class WrapData(Main):
    """
    """
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        super().__init__(self, *args, **kwargs)
        self.file_df         = pd.read_csv(self.filenamepath + self.const_fileext)
        self.frame_df        = pd.read_csv(self.filenamepath + self.const_frameext)
        

    def post_process(self, *args, **kwargs):
        """
        """
        
        fps    = self.file_df.iloc[0]['framerateencoded']
        frames = self.file_df.iloc[0]['totalframes']
        
        df = self.frame_df.loc[(self.frame_df['objectcount'] != 0.0)][['framenumber','framegroup']]
        
        self.frame_df['writeflag'] = 0
        self.frame_df['newframenumber'] = 0

        if len(df):
            grp = list(range_group(df['framegroup'].tolist()))
            for cnt in range(len(grp)):
                sgrp, egrp = grp[cnt]
                sframe = self.frame_df.loc[(self.frame_df['framegroup'] == sgrp)][['framenumber']].min()[0]
                eframe = self.frame_df.loc[(self.frame_df['framegroup'] == egrp)][['framenumber']].max()[0]
                newsframe = max(sframe - (fps * self.const_bufferseconds), 1) 
                neweframe = min(eframe + (fps * self.const_bufferseconds), frames)
                self.frame_df.loc[newsframe - 1: sframe - 1, 'writeflag'] = 2
                self.frame_df.loc[sframe - 1: eframe, 'writeflag'] = 1
                self.frame_df.loc[eframe: neweframe - 1, 'writeflag'] = 2
            
            df = self.frame_df.loc[(self.frame_df['writeflag'] != 0.0)].copy()
            df['newframenumber'] = 0
            df['newframenumber'] = df.groupby('newframenumber').cumcount() + 1
            self.frame_df.update(df)          
                          
        
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
        self.outputfilesize = '0 KB'
        
        if len(df):
            opname = self.file_df.iloc[0]['outputfilenamepath']     
            self.outputfilesize = Path(opname).stat().st_size // 1024
            self.outputfilesize = '{0} KB'.format(self.outputfilesize)            
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
        
        self.file_df['filepartial']       = self.file_df['filepartial'].apply(lambda x: 'Y' if x else 'N')
        self.file_df['outputfilesize']    = self.outputfilesize
        self.file_df['totalruntime']      = round(self.log_df['duration'].sum(), 2)
        
        super().save(self, *args, **kwargs)     