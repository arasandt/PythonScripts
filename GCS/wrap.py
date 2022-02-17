#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import pandas as pd

from mainclass import Main
from utils import range_group

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

    
    def save(self, *args, **kwargs):
        """
        """
        
        super().save(self, *args, **kwargs)                    