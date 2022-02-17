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

import os, sys
import pandas as pd

from datetime import datetime
from collections import defaultdict

from core import utils


class Main:
    """
    """
    
    def __init__(self, *args, **kwargs):    
        """
        Initializes all instance variable on the video
        
        """
        
        self.processstarttime = datetime.utcnow()
        self.__dict__.update(kwargs)
        
        classlocation = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
        self.settings = utils.load_constants(self, classlocation, self.debug)

        if self.loadfiles:
            self.log_df  = pd.read_csv(self.filenamepath + self.const_logext)
       
        
    def save_log(self, *args, **kwargs):
        """
        """
        
        self.processendtime = datetime.utcnow()

        temp_dict = defaultdict(dict)
        temp_dict['log'] =         {'process'           : self.process, 
                                    'starttime'         : self.processstarttime.strftime('%m/%d/%Y %I:%M:%S %p'),
                                    'endtime'           : self.processendtime.strftime('%m/%d/%Y %I:%M:%S %p'),
                                    'duration'          : round((self.processendtime - self.processstarttime).total_seconds(), 2),
                                    'settings'          : self.settings,
                                    }
        df = pd.DataFrame.from_dict(temp_dict, 'index')
        
        if self.loadfiles:
            self.log_df = pd.concat([self.log_df, df])
            self.log_df.drop_duplicates(subset=['process'], inplace=True, keep='last')
            self.log_df.to_csv(self.filenamepath + self.const_logext, header=True, sep=',', index=False)
        else:
            df.to_csv(self.filenamepath + self.const_logext, header=True, sep=',', index=False)
        

    def save(self, *args, **kwargs):
        """
        """
        
        if hasattr(self, 'file_df'):
            self.file_df.to_csv(self.filenamepath + self.const_fileext, header=True, sep=',', index=False)

        if hasattr(self, 'frame_df'):
            self.frame_df.to_csv(self.filenamepath + self.const_frameext, header=True, sep=',', index=False)        
        
        if hasattr(self, 'framedetails_df'):
            #if len(self.framedetails_df):
            self.framedetails_df.to_csv(self.filenamepath + self.const_framedetailsext, header=True, sep=',', index=False)        

        self.save_log(self, *args, **kwargs)
        
                