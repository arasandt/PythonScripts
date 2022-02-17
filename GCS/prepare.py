#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd

from mainclass import Main

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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