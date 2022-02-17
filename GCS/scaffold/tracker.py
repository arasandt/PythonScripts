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

from datetime import datetime, timedelta
from dateutil import tz
from pathlib import Path

from core.mainclass import Main
from core import utils
from core.utils import Progress


class TrackerObject(Main):
    """
    """

    OPENCV_OBJECT_TRACKERS = {
                        		"csrt": cv2.TrackerCSRT_create,
                        		"kcf": cv2.TrackerKCF_create,
                        		"boosting": cv2.TrackerBoosting_create,
                        		"mil": cv2.TrackerMIL_create,
                        		"tld": cv2.TrackerTLD_create,
                        		"medianflow": cv2.TrackerMedianFlow_create,
                        		"mosse": cv2.TrackerMOSSE_create
                             }
                                       
    def __init__(self, *args, **kwargs):
        """
        """
        
        super().__init__(self, *args, **kwargs)
        self.file_df                = pd.read_csv(self.filenamepath + self.const_fileext)
        self.frame_df               = pd.read_csv(self.filenamepath + self.const_frameext)
        try:
            self.framedetails_df        = pd.read_csv(self.filenamepath + self.const_framedetailsext)
        except pd.errors.EmptyDataError:
            self.framedetails_df = pd.DataFrame()
        
        self.const_videoextdef   = '.{0}.'.format(self.const_videocompressiondef)        
        self.const_ffmpegextdef  = '.{0}.'.format(self.const_ffmpegcompressiondef)             
        
        self.file_df = self.file_df.fillna("")
        self.outputfilenamepath = self.file_df.iloc[0]['outputfilenamepath']

        if not self.outputfilenamepath or not Path(self.outputfilenamepath).exists():
            print('Output video file {0} not found'.format(self.outputfilenamepath), flush=True)        
        
        self.starttimeencoded = self.file_df.iloc[0]['starttimeencoded']
        self.fps    = self.file_df.iloc[0]['framerateencoded']
        self.width  = self.file_df.iloc[0]['widthencoded']
        self.height = self.file_df.iloc[0]['heightencoded']        
    
    
    def perform_tracking(self, *args, **kwargs):
        """
        """        
        
        activity = 'Applying OpenCV Tracker ({0})'.format(self.const_trackeralgo)
        
        def get_image_for_frame_number(selected_frames, frame_number):
            for frame in selected_frames:
                f, g, i = frame
                if f == frame_number:
                    return i
                
        df = self.frame_df.loc[(self.frame_df['writeflag'] != 0.0)][['framenumber','newframenumber','framegroup','objectcount']]
        
        print('Selecting {0} image(s)'.format(len(df)), flush=True)
        
        selected_frames = utils.get_images(self.outputfilenamepath, 
                                           list(zip(df['newframenumber'],df['framegroup']))) 

        tracklets = []
        temp_dict = {}
        tracker_count = 0
        tup = 0

        with Progress(activity, len(df)) as pbar:
            for index, row in df.iterrows():
                img = get_image_for_frame_number(selected_frames, row['newframenumber'])
                if row['objectcount'] == 0:
                    if tracklets:
                        for cnt, track in enumerate(tracklets):
                            (success, box) = track.update(img)
                            box = list(map(int,box))
                            if success:
                                temp_dict[tracker_count] = {
                                                               'framenumber' : row['framenumber'],
                                                               'framegroup'  : row['framegroup'],
                                                               'object'      : 'tracker',
                                                               'sequence'    : cnt,
                                                               'box'         : str(box),
                                                               'confidence'  : 1,
                                                           }
                                tracker_count += 1
                else:
                    tracklets = [] # reset trackers when person was found using person detection
                    box_df = self.framedetails_df.loc[(self.framedetails_df['framenumber'] == row['framenumber']) & (self.framedetails_df['framenumber'] != 'tracker')][['box']]
                    for box_row in list(box_df['box']):
                        tracklet = self.OPENCV_OBJECT_TRACKERS[self.const_trackeralgo.lower()]()
                        tracklet.init(img, tuple(eval(box_row)))
                        tracklets.append(tracklet)
                tup += 1
                pbar.update(tup)
        
        df_tracker = pd.DataFrame.from_dict(temp_dict, 'index')
        
        self.framedetails_df = self.framedetails_df.loc[(self.framedetails_df['object'] != 'tracker')]
        self.framedetails_df =  self.framedetails_df.append(df_tracker, ignore_index = True, sort=True)
        

    def write_result_frames(self, *args, **kwargs):
        """
        """
        
        df = self.framedetails_df.copy()
        df['boxconfidence'] = df.apply(lambda row: str(tuple((eval(row.box), row.confidence))), axis=1)
        df_grp = df.groupby(['framenumber'])['boxconfidence'].apply(lambda x: '|'.join(x))

        df = self.frame_df.loc[(self.frame_df['writeflag'] != 0.0)][['framenumber','newframenumber','framegroup']]
        df = pd.merge(df, df_grp, on='framenumber', how='left').fillna(0)
        df.drop(['framenumber'], axis=1, inplace=True, errors='ignore')
       
        if len(df):
                inputfilename = self.outputfilenamepath + '.temp.' + self.const_videocontainerdef
                out = cv2.VideoWriter(inputfilename, cv2.VideoWriter_fourcc(*self.const_videocompressiondef.upper()), self.fps, (self.width,self.height))    
                
                print('Selecting {0} image(s)'.format(len(df)), flush=True)
                
                utils.get_images(self.outputfilenamepath, 
                                 list(zip(df['newframenumber'],df['framegroup'])), 
                                 outwrite=out,
                                 applyresult=list(df['boxconfidence']))
                
                out.release()
                
                self.resultfilenamepath = self.outputfilenamepath +  '.result.' + self.const_ffmegcontainerdef
                
                out, _ = (
                            ffmpeg
                            .input(inputfilename)
                            .output(self.resultfilenamepath, crf=self.const_ffmpegcrfdef, vcodec=self.const_ffmpegcompressiondef)
                            .overwrite_output()
                            .run()
                         )
                
                # remove the temp video file    
                os.remove(inputfilename)    

                
    def create_subtitles(self, *args, **kwargs):
        """
        """
        
        starttime = datetime.strptime('00:00:00', "%H:%M:%S")  
        secsCount = 0 
        timeformat = '%m/%d/%Y %I:%M:%S %p %Z'
        from_zone, to_zone = tz.gettz('UTC'), tz.gettz(self.const_subtitletimezone)
        
        ofile = self.outputfilenamepath
        
        srtfilename = ofile + self.const_srtext
        vstarttime = self.starttimeencoded
        vstarttime = datetime.strptime(vstarttime, "%m/%d/%Y %I:%M:%S %p")
        vstarttime = vstarttime.replace(tzinfo=from_zone)
        vstarttime = vstarttime.astimezone(to_zone)
        
        allframes = self.frame_df[self.frame_df['writeflag'] != 0.0]['framenumber'].tolist()

        with open(srtfilename, "w+") as srtout:
            
            for i in range(0, len(allframes), self.fps):
                srtout.write('{0}\n'.format(str(secsCount)))
                
                srtout.write('{0},000-->{0},999\n'.format(starttime.strftime('%H:%M:%S')))
                starttime += timedelta(seconds = 1) 
                
                nexttime = vstarttime + timedelta(seconds = (allframes[i] // self.fps))
                srtout.write('{0}\n'.format(nexttime.strftime(timeformat)))

                srtout.write('\n')
                secsCount += 1                