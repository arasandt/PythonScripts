#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains all utilities
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import yaml, os

from progressbar import Bar, Percentage, ProgressBar, Timer
from collections import OrderedDict


class Progress(): 
    """
    """
    
    def __init__(self, *args, **kwargs): 
        self.name = args[0]
        if self.name:
            self.name = "{0} ".format(self.name)
        self.total = args[1]
          
    def __enter__(self): 
        self.pbar = ProgressBar(widgets=[self.name, Percentage(), Bar(), Timer(),], maxval=self.total).start()
        return self.pbar
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        self.pbar.finish()
    
    def update(self,count):
        self.pbar.update(count)
    
    
def range_group(seq):
    start, end = seq[0], seq[0]
    count = start
    for item in seq:
        if not count == item:
            yield start, end
            start, end = item, item
            count = item
        end = item
        count += 1
    yield start, end


def load_constants(obj, classlocation, debug):
    """
    """
    
    const_filename = obj.__class__.__name__ + ".yaml"
    const_filename = os.path.join(classlocation, const_filename)
    
    if os.path.exists(const_filename):
        with open(const_filename, 'r') as file:
            yaml_details = yaml.load(file, Loader=yaml.Loader)
            yaml_parms = yaml_details['parameters']
            [setattr(obj, 'const_' + i, j) for i, j in yaml_parms.items()]
            obj_items = {k:v for k,v in obj.__dict__.items() if 'const_' in k}
            
            if debug:
                max_obj_key_length = max([len(k) for k in obj_items.keys()])
                for key, value in obj_items.items():
                    print('{0: <{1}} : {2}'.format(key,max_obj_key_length,value))
            
            #return obj_items
            return None

def get_process_to_run(proceed, process, procname):
    """
    """
    
    if process is None or not proceed:
        selected_process = OrderedDict({k:v for k,v in procname.items() if v[0]})
        return selected_process, process        
    else:
        selected_process = OrderedDict({k:v for k,v in procname.items() if v[0]})
        if process in list(selected_process.keys()):
            idx = list(selected_process.keys())
            idx = idx.index(process)
            t_dict = OrderedDict({})
            for count, item in enumerate(selected_process.items()):
                if count >= idx:
                    k,v = item
                    t_dict[k] = v
            selected_process = t_dict
        return selected_process, None

    
    
    