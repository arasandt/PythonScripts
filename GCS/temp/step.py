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

import sys, traceback

from collections import OrderedDict
from datetime import datetime

from process import *


class ProcessStep:
    """
    """
    
    procname = OrderedDict({ 'metadata'      : ['extract_clean_metadata', ],
                             'scan'          : ['full_scan',              ],
                             'prepare'       : ['prepare_data_for_model', ],
                             'model'         : ['run_model',              ],
                             'wrap'          : ['wrap_data_from_model',   ],
                             'output'        : ['generate_output',        ],
                             'validate'      : ['validate_output',        ],
                             #'tracker'       : ['run_tracker',            ],                             
                          })    
    
    adhocprocname = OrderedDict({ 'tracker'        : ['run_tracker',            ],
                                  'subtitle'       : ['create_subtitle',        ],
                                  'boundingbox'    : ['show_boundingbox',        ],
                               })   
    
    
    def __init__(self, *args, **kwargs):
        """
        """
        
        self.kwargs = {}
        self.__dict__.update(kwargs)
        self.saved_kwargs = kwargs
       
    
    def _call_process(self, *args, **kwargs):  
        """
        """

        func_name = args[0]

        try:
            eval(func_name + '(*args,**kwargs)')
        except Exception as error:
            logger.exception(error)    
            error_type, error_message, error_traceback = sys.exc_info()
            entire_error = ''.join(traceback.format_tb(error_traceback)) + '{0}: {1}'.format(error_type.__name__, error_message)
            print(datetime.utcnow(), flush=True)
            return (False, entire_error)
        
        return (True, 'Success')    
    
    
    def execute(self):
        """
        """
        
        kwargs = self.kwargs
        fname = kwargs.get('filenamepath',"")
        return_result = OrderedDict({proc:(False, None) for proc,value in self.selected_process.items()})
        for proc, args in self.selected_process.items():
            print('{2} @ {1} : Started  << {0: ^15} >>'.format(proc.upper(), datetime.utcnow(), fname), flush=True)
            kwargs['process'] = proc
            ret_res, log_info = self._call_process(*args, **kwargs)
            kwargs = self.kwargs
            print('{2} @ {1} : Finished << {0: ^15} >>\n'.format(proc.upper(), datetime.utcnow(), fname), flush=True)
            return_result[proc] = (ret_res, log_info)
            if not ret_res:
                break
        
        return return_result
        
    
    def get_processes_with_args(self, *args, **kwargs):
        """
        """
        
        if self.process is None:
            self.selected_process = self.procname
            self.resume = False
            self.framenum = ()
        else:
            self.process = self.process.lower()
            if self.process in self.procname.keys():
                if self.resume:
                    self.framenum = ()
                    self.selected_process = self.procname
                    if self.process in list(self.selected_process.keys()):
                        idx = list(self.selected_process.keys())
                        idx = idx.index(self.process)
                        t_dict = OrderedDict({})
                        for count, item in enumerate(self.selected_process.items()):
                            if count >= idx:
                                k,v = item
                                t_dict[k] = v
                        self.selected_process = t_dict
                    self.process = None
                else:
                    self.selected_process = OrderedDict({})
                    self.selected_process[self.process] = self.procname[self.process]
            else:
                self.selected_process = OrderedDict({})
                if self.process in self.adhocprocname.keys():
                    self.selected_process[self.process] = self.adhocprocname[self.process]                    
                else:
                    raise Exception('Process {0} not found'.format(self.process))
        
        self.kwargs = {k:getattr(self,k) for k,v in self.saved_kwargs.items()}
            
            
            
            
            
            
            
            
            
            
            
