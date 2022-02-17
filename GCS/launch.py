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

import click, sys
import os

from datetime import datetime
from pathlib import Path

from core.utils import RedirectStdStreams
from core import step


def main_process(**kwargs):
    """
    """

    args = []
    kwargs.pop('ctx', None)
    kwargs['filenamepath'] = kwargs.get('filenamepath', None)
    kwargs['process']      = kwargs.get('process', None)
    kwargs['modelfolder']  = kwargs.get('modelfolder', None)
    kwargs['framenum']     = kwargs.get('framenum', ())
    kwargs['resume']       = kwargs.get('resume', False)
    kwargs['debug']        = kwargs.get('debug', False)
    kwargs['loadfiles']    = kwargs.get('loadfiles', True)
    kwargs['dbfps']        = kwargs.get('dbfps', 10)
    
    trigger = kwargs.get('trigger', 'NA')

    if not kwargs['debug']:
        print('{3}#{2} >>> {0} {1}'.format(kwargs['filenamepath'], kwargs['dbfps'], trigger, os.getpid()), flush=True)
        log_file = 'mainlog.log'
        if Path(log_file).exists():
            if Path(log_file).stat().st_size > 5242880: # backup if greater than 5 MB
                os.rename(log_file, log_file + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        with open(log_file,'a') as ml:
            ml.write('{3}#{2} >>> {0} {1} start\n'.format(datetime.utcnow(), kwargs['filenamepath'], trigger, os.getpid()))
        f = open(kwargs['filenamepath'] + '.runlog', 'w')
    else:
        f = sys.stdout
            
    with RedirectStdStreams(stdout=f, stderr=f):
        print('Input Arguments\n{0}'.format(kwargs), flush=True)
        stepfunc = step.ProcessStep(*args, **kwargs)
        stepfunc.get_processes_with_args()
        print('Processed Arguments\n{0}'.format(stepfunc.kwargs), flush=True)
        return_result = stepfunc.execute()

    with open('mainlog.log','a') as ml:
        ml.write('{3}#{2} >>> {0} {1} end\n'.format(datetime.utcnow(), kwargs['filenamepath'], trigger, os.getpid()))
        

    return return_result
   

@click.command()
@click.option('-f', '--filenamepath', required=True, type=str, help='Input file path name to be processed')
@click.option('-p', '--process', type=str, help='Process to be applied on the input file name')
@click.option('-m', '--modelfolder', type=str, help='Model to apply on frames')
@click.option('-n', '--framenum', type=int, multiple=True, help='Frame Number on which model should be applied')
@click.option('--resume', is_flag=True)
@click.option('--debug', is_flag=True)
@click.pass_context
def main_function(ctx, filenamepath, process, modelfolder, framenum, resume, debug):
    """
    """
    
    try:
        main_process(ctx=ctx, 
                     filenamepath=filenamepath, 
                     process=process, 
                     modelfolder=modelfolder, 
                     framenum=framenum, 
                     resume=resume,
                     debug=debug,
                     )    
    except Exception as error:
        logger.exception(error)    
        print(datetime.utcnow(), flush=True)        
        raise
    
    
if __name__ == '__main__':
    """
    """

    main_function()
    