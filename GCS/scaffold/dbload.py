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

import os, glob
import json, traceback
import sys

from shutil import move
from pathlib import Path

from LenelJobDAO import LenelJobDAO
from core.utils import sendemail


def call_exception(fromfile, tofile, error, return_res, newfilepath):
    """
    """
    
    move(fromfile, tofile)

    logger.exception(error)            
    error_type, error_message, error_traceback = sys.exc_info()
    entire_error = ''.join(traceback.format_tb(error_traceback)) + '{0}: {1}'.format(error_type.__name__, error_message)

    return_res['error'] = entire_error
    write_file(tofile, return_res)
    sendemail('{0} - {1} failed'.format(newfilepath, return_res['fileid']))
    
    
def write_file(tofile, return_res):
    """
    """
    
    runlogs_res = json.dumps(return_res)
    with open(tofile,"w") as triggerfileOut:
        triggerfileOut.write(runlogs_res)      


def call_success(fromfile, tofile, return_res):
    """
    """
    
    move(fromfile, tofile)
    
    return_res['error'] = 'Success'
    write_file(tofile, return_res)    
    

def call_read(file, namereplace):
    """
    """
    
    print('Loading DB for {0}'.format(file))
    with open(file,"r") as filein:
        filecontent = filein.readline() 
    
    tail = Path(file).name
    fileid = tail.replace(namereplace,'')
    
    return json.loads(filecontent), fileid


def load_db():
    """
    """

    Outtriggerpath = os.path.join(os.getcwd(),"output")
    
    ldao = LenelJobDAO()
    
    for file in glob.glob(Outtriggerpath + '\*.trg.error'):
        return_res, fileid = call_read(file, '.trg.error')
        newfilepath = return_res['filecontent'].split('|')[0].replace('.lnr','.h264')
        try:
            ldao.uploadFileInfoLog(newfilepath, fileid)
            ldao.update_file_compress_status(fileid,'FAILED')
        except Exception as error:
            call_exception(file, file.replace('.dbready','.dberror'), error, return_res, newfilepath)
        
    
    for file in glob.glob(Outtriggerpath + '\*.trg.dbready'):
        return_res, fileid = call_read(file, '.trg.dbready')
        newfilepath = return_res['filecontent'].split('|')[0].replace('.lnr','.h264')
        try:
            ldao.uploadFileInfo(newfilepath, fileid)
            ldao.uploadFramesInfo(newfilepath, fileid)
            ldao.uploadFrameDetailsInfo(newfilepath, fileid)
            ldao.uploadFileInfoLog(newfilepath, fileid)
            ldao.update_file_compress_status(fileid,'SUCCESS')        
            call_success(file, file.replace('.dbready','.dbcomplete'), return_res)
        except Exception as error:
            call_exception(file, file.replace('.dbready','.dberror'), error, return_res, newfilepath)
            
            
if __name__ == '__main__':
    """
    """
    
    load_db()

        
    
    
    