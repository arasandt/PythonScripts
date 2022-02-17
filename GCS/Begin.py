#!/usr/bin/python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

"""
python Begin.py "Met1 NW Entrance.lnr" scan && python Process.py "Met1 NW Entrance.lnr" prepare && python Process.py "Met1 NW Entrance.lnr" runmodel YOLOv3-spp && python Process.py "Met1 NW Entrance.lnr" output && python Process.py "Met1 NW Entrance.lnr" validate
python Begin.py "Met1 Front to Parking.lnr" scan && python Process.py "Met1 Front to Parking.lnr" prepare && python Process.py "Met1 Front to Parking.lnr" runmodel YOLOv3-spp && python Process.py "Met1 Front to Parking.lnr" output && python Process.py "Met1 Front to Parking.lnr" validate
python Begin.py "Met1MDFOutside.lnr" scan && python Process.py "Met1MDFOutside.lnr" prepare && python Process.py "Met1MDFOutside.lnr" runmodel  YOLOv3-spp && python Process.py "Met1MDFOutside.lnr" output && python Process.py "Met1MDFOutside.lnr" validate
python Begin.py "Met1 Juice Bar.lnr" scan && python Process.py "Met1 Juice Bar.lnr" prepare && python Process.py "Met1 Juice Bar.lnr" runmodel  YOLOv3-spp && python Process.py "Met1 Juice Bar.lnr" output && python Process.py "Met1 Juice Bar.lnr" validate
"""

import time, sys
import VideoManager

from datetime import datetime


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def scan(*args, **kwargs):
    """
    """
    
    video = VideoManager.VideoFile(*args, **kwargs)
    video.process_frames(*args, **kwargs)
    video.save_attributes(*args, **kwargs)

    
if __name__ == '__main__':
    """
    """
    
    start = time.time()

    if len(sys.argv) <= 1:
        print('Provide input file name and process name!!!')
    else:
        videoname = sys.argv[1]
        process = sys.argv[2]   
        
    procname = { 'scan'     : '(videoname, process)',
                }
    
    print('Process **** {0} **** started  - {1}'.format(process,datetime.now()))
    try:
        eval(process + procname[process])
    except Exception as error:
        logger.exception(error)
    print('Process **** {0} **** complete - {1}'.format(process,datetime.now()))
    
    #analyze_video('Met1MDFOutside.lnr') # big
    #analyze_video('Met1 NW Entrance.lnr') 
    #analyze_video('Met1 Front to Parking.lnr') # small
    #analyze_video('Met1 Juice Bar.lnr')
    
    end = time.time()
    print('Time Taken : {0} seconds\n'.format(end - start))
    
