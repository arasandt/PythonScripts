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

from pathlib import Path

from core.mainclass import Main
from core import utils
from core.utils import Progress


class MainModel(Main):
    """
    """
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        super().__init__(self, *args, **kwargs)
        self.file_df = pd.read_csv(self.filenamepath + self.const_fileext)
        self.frame_df = pd.read_csv(self.filenamepath + self.const_frameext)

        self.file_df = self.file_df.fillna("")
        self.outputfilenamepath = self.file_df.iloc[0]['outputfilenamepath']
        if not self.outputfilenamepath or not Path(self.outputfilenamepath).exists():
            raise Exception('Input video file {0} not found'.format(self.outputfilenamepath))   
            
        self.validate_model()

    
    def validate_model(self, *args, **kwargs):
        """
        """
        
        if self.modelfolder is None:
            raise Exception('Model folder {0} is incorrect'.format(self.modelfolder))        
        
        p = Path(self.modelfolder)
        if not p.is_dir():
            raise Exception('Model folder {0} does not exists'.format(self.modelfolder))
        
        pkg_dir, pkg_base = p.parent, p.name
        pkg_files = [Path(i).stem for i in os.listdir(self.modelfolder) if '.py' in i.lower()]
        pkg_files = [i for i in pkg_files if i.lower() == pkg_base.lower()]

        if not pkg_files:
            raise Exception('Model folder {0} does not contain {1}.py file to import'.format(self.modelfolder, pkg_base))

        pkg_file = pkg_files[0]
        pkg = os.path.join(pkg_dir, pkg_file)
        self.modelfolder = pkg
        
        from importlib import import_module
        sys.path.append(self.modelfolder)
        self.model_pkg = import_module(Path(Path(self.modelfolder).resolve()).name)
        

    def select_frames_for_model(self, *args, **kwargs):
        """
        """
        
        if self.framenum:
            # select only the frame which was passed as input in framenum field
            roi_frames = self.frame_df.loc[(self.frame_df['framenumber'].isin(list(self.framenum)))]
        else:
            # select only frames which have motion and matches with framerank.
            
            roi_frames = self.frame_df.loc[(self.frame_df['writeflag'] != 0.0)][['framenumber','framegroup','newframenumber']]
            
            
        roi_frames = roi_frames[1240:1300]
        
        #zip the columns
        self.model_frames = list(zip(list(roi_frames['newframenumber']),list(roi_frames['framegroup'])))
        
        print('Selecting {0} image(s)'.format(len(self.model_frames)), flush=True)
        
        self.model_frames = utils.get_images(self.outputfilenamepath, 
                                             self.model_frames)   

        for cnt in range(len(self.model_frames)):
            self.model_frames[cnt] = (list(roi_frames['framenumber'])[cnt], self.model_frames[cnt][1],self.model_frames[cnt][2])
            
        
       
    def apply_model_on_frames(self, *args, **kwargs):
        """
        """
        
        activity = 'Applying Model ({0})'.format(self.modelfolder)
        
        facedet = self.model_pkg.ModelObject(self.outputfilenamepath, self.modelfolder, debug=self.debug)
        if self.settings is not None and facedet.settings is not None:
            self.settings.update(facedet.settings)
        
        model_success = 0
        model_result_counter = 0
        temp_dict = {}
        
        with Progress(activity, len(self.model_frames)) as pbar:
        
            for tup in range(len(self.model_frames)):
                frame, group, image = self.model_frames[tup]
                facedet.image = image
                facedet.framenumber = frame 
                result, _ = facedet.detect()
                   
                self.model_frames[tup] = (frame, group, result)
                
                if result != -1:
                    model_success += 1
                    for cnt, val in enumerate(result):
                        coor, confi, classname = val
                        temp_dict[model_result_counter] = {
                                                           'framenumber' : frame,
                                                           'framegroup'  : group,
                                                           'object'      : classname,
                                                           'sequence'    : cnt,
                                                           'box'         : str(coor),
                                                           'confidence'  : confi,
                                                          }
                        model_result_counter += 1
                
                pbar.update(tup + 1)
             
        print('Model successful on {0} image(s) with {1} hit(s)'.format(model_success, len(temp_dict)), flush=True)
        
        self.framedetails_df = pd.DataFrame.from_dict(temp_dict, 'index')
                
        
    def save(self, *args, **kwargs):
        """
        """
        
        super().save(self, *args, **kwargs)        