#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import click, Utils
import ModelManager, VideoManager
import sys, traceback
 
from datetime import datetime
from collections import OrderedDict


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def prepare(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.prepare_data(*args, **kwargs)
    
    video.save_attributes(*args, **kwargs)

    
def runmodel(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)

    video.validate_model(*args, **kwargs)
    video.select_frames_for_model(*args, **kwargs)

    if video.framenum:
        video.apply_model_on_singled_frame(*args, **kwargs)
    else:
        video.apply_model_on_frames(*args, **kwargs)

        video.save_attributes(*args, **kwargs)
    

def rerunmodel(*args, **kwargs):
    """
    """
    
    runmodel(*args, **kwargs)


def tracker(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.perform_tracking(*args, **kwargs)
    
    video.save_attributes(*args, **kwargs)
    

def output(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.write_model_frames(*args, **kwargs)
    
    video.save_attributes(*args, **kwargs)    


def outputresult(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.write_model_frames_with_result(*args, **kwargs)
    
    #video.save_attributes(*args, **kwargs)
  
 
def validate(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.validate_inp_outp(*args, **kwargs)
    
    video.save_attributes(*args, **kwargs)


def subtitle(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.create_subtitles(*args, **kwargs)
    
    video.save_attributes(*args, **kwargs)
    

def dbextracts(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.create_db_extracts(*args, **kwargs)
    
    video.save_attributes(*args, **kwargs)
    

def scan(*args, **kwargs):
    """
    """
    
    video = VideoManager.VideoFile(*args, **kwargs)
    video.pull_metadata(*args, **kwargs)
    video.process_frames(*args, **kwargs)
    video.save_attributes(*args, **kwargs)
    

def wrap(*args, **kwargs):
    """
    """
    
    video = ModelManager.ManagerObject(*args, **kwargs)
    video.load_metadata(*args, **kwargs)
    
    video.post_process(*args, **kwargs)
    
    video.save_attributes(*args, **kwargs)
    

def call_process(procname, *args):  
    args = list(args)
    process = args[1] = args[1].lower()
    args = tuple(args)
    print('Process **** {0: ^15} **** started  - {1}'.format(process, datetime.utcnow()))
    try:
        eval(process + procname[process][1])
    except Exception as error:
        logger.exception(error)    
        error_type, error_message, error_traceback = sys.exc_info()
        entire_error = ''.join(traceback.format_tb(error_traceback)) + '{0}: {1}'.format(error_type.__name__, error_message)
        print(datetime.utcnow())
        return (False, entire_error)
    print('Process **** {0: ^15} **** complete - {1}'.format(process, datetime.utcnow()))
    return (True, 'Success')
    

def main_process(**kwargs):
    """
    """
    
    #frame number is not valid if all steps have to be run
    ctx = kwargs.get('ctx', None)
    filepath = kwargs.get('filepath', None)
    process = kwargs.get('process', None)
    modelfolder = kwargs.get('modelfolder', None)
    framenum = kwargs.get('framenum', ())
    proceed = kwargs.get('proceed', False)
    debug = kwargs.get('debug', False)
    
    if process is None:
        framenum = ()
        
    if len(framenum) != 0:
        proceed = False
    
    args = [filepath, process, modelfolder, framenum, proceed, debug]
    print(args)

    procname = OrderedDict({ 'scan'          : [1,'(*args)'],
                             'prepare'       : [1,'(*args)'],
                             'runmodel'      : [1,'(*args)'],
                             'tracker'       : [1,'(*args)'],                             
                             'rerunmodel'    : [0,'(*args, rerun=True)'],                 
                             'wrap'          : [1,'(*args)'],
                             'output'        : [1,'(*args)'],
                             'outputresult'  : [0,'(*args)'],                             
                             'validate'      : [1,'(*args)'],
                             'subtitle'      : [1,'(*args)'],
                             'dbextracts'    : [1,'(*args)'],
                          })
    
    selected_process, process = Utils.get_process_to_run(proceed, process, procname)

    if process is None:
        return_result = OrderedDict({process:(False, None) for process,value in selected_process.items()})
        for process, value in selected_process.items():
                args[1] = process
                ret_res, log_info = call_process(procname, *args)
                return_result[process] = (ret_res, log_info)
                if not ret_res:
                    break
                    
    else:
        ret_res, log_info = call_process(procname, *args)
        return_result = {process:(ret_res, log_info)}
    
    return return_result
    
    
@click.command()
@click.option('-f', '--filepath', required=True, type=str, help='Input file path name to be processed')
@click.option('-p', '--process', type=str, help='Process to be applied on the Input file name')
@click.option('-m', '--modelfolder', type=str, help='Model to apply on frames')
@click.option('-n', '--framenum', type=int, multiple=True, help='Frame Number on which Model should be applied')
@click.option('--proceed', is_flag=True)
@click.option('--debug', is_flag=True)
@click.pass_context
def main_function(ctx, filepath, process, modelfolder, framenum, proceed, debug):
    """
    """
    
    try:
        main_process(ctx=ctx, 
                     filepath=filepath, 
                     process=process, 
                     modelfolder=modelfolder, 
                     framenum=framenum, 
                     proceed=proceed,
                     debug=debug,
                     )    
    except Exception as error:
        logger.exception(error)    
        print(datetime.utcnow())        
        raise
    
    
if __name__ == '__main__':
    """
    """

    main_function()
    

    