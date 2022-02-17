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

import json, os
import time, glob
import random

import launch

from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler


class EventHandler(PatternMatchingEventHandler):
    """
    """
      
    
    def on_created(self, event): 
        """
        """
        time.sleep(round(random.uniform(0.1, 3.0), 2))
        if Path(event.src_path).exists():
            self.executeModel(event.src_path)
        

    def executeModel(self, srcpath):   
        """
        """
        
        Outtriggerpath = os.path.join(os.getcwd(),"output")
        tail = Path(srcpath).name        
        fileid = Path(tail).stem
        if fileid.isnumeric():
            newfile = srcpath + ".processing"
            #time.sleep(1)
            try:
                os.rename(srcpath, newfile)
            except (FileNotFoundError, OSError):
                return
            
            with open(newfile,"r") as filein:
                filecontent = filein.readline()
            filepath, fps = filecontent.split('|')
            
            flag, newfilepath = self.createWorkingDir(filepath, 'Analytics')
            
            if flag:                
                return_res = launch.main_process(filenamepath=newfilepath, modelfolder='models/YOLOv3_spp', dbfps=int(fps), trigger=tail)                
            else:
                return_res = {"copy": [False, 'Failed creating working file {0}'.format(newfilepath)]}
            
            failed_ones = {i:j for i,j in return_res.items() if not j[0]}

            if len(failed_ones):
                res = '.error'
            else:
                try:
                    os.remove(newfilepath)
                except (FileNotFoundError, OSError):
                    pass
                res = '.dbready'
                
            filenameout = os.path.join(Outtriggerpath, tail + res)
            return_res['filecontent'] = filecontent
            return_res['fileid'] = fileid
            runlogs_res = json.dumps(return_res)
            with open(filenameout,"w") as triggerfileOut:
                triggerfileOut.write(runlogs_res)    

            try:
                os.remove(newfile)
            except (FileNotFoundError, OSError):
                pass


    def createWorkingDir(self, inputpath, mainfolder):
        p = Path(inputpath)
        part = list(p.parts)
        part[-1] = part[-1].replace('.lnr','.h264')
        part.insert(1, mainfolder)
        newpath = os.path.join(*part)
        os.system('echo f | xcopy /y "{source}" "{target}" > nul'.format(source=inputpath, target=newpath))
        if not Path(newpath).exists(): 
            return False, newpath
        return True, newpath


if __name__ == '__main__':
    """
    """

    InputTriggerpath = os.path.join(os.getcwd(),"input")
    
    event_handler = EventHandler(patterns=["*.trg"])  
    
    # process existing files
    for file in glob.glob(InputTriggerpath + '\*.*'):
        if Path(file).exists():
            event_handler.executeModel(file)
    
    observer = Observer()    
    observer.schedule(event_handler, InputTriggerpath, recursive=False)    
    observer.start()  
    print('Watching...')
    try:
        while True:
            time.sleep(1)          
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    
    