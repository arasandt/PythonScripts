#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains all utilities
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import yaml, os
import cv2, contextlib
import sys, smtplib

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from collections import OrderedDict
from datetime import datetime, timedelta
from dateutil import tz
from PIL import Image
from pathlib import Path

from progressbar import Bar, Percentage, ProgressBar, Timer


class Progress(): 
    """
    """
    
    def __init__(self, *args, **kwargs): 
        self.name = args[0]
        if self.name:
            self.name = "{0} ".format(self.name)
        self.total = args[1]
          
    def __enter__(self): 
        #self.pbar = ProgressBar(widgets=[self.name, Percentage(), Bar(marker=u"\u2588"), Timer(),], maxval=self.total).start()
        self.pbar = ProgressBar(widgets=[self.name, Percentage(), Bar(marker='#'), Timer(),], maxval=self.total).start()
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
    
    if Path(const_filename).exists():
        with open(const_filename, 'r') as file:
            yaml_details = yaml.load(file, Loader=yaml.Loader)
            yaml_parms = yaml_details['parameters']
            [setattr(obj, 'const_' + i, j) for i, j in yaml_parms.items()]
            obj_items = {k:v for k,v in obj.__dict__.items() if 'const_' in k}
            #max_obj_key_length = max([len(k) for k in obj_items.keys()])
            
            if debug:
                x = []
                for key, value in obj_items.items():
                    x.append('{0: <{1}} : {2}'.format(key,1,value))
                print('CONSTANTS {0} >>>> {1}'.format(obj.__class__.__name__, ', '.join(x)), flush=True)
            
            return obj_items


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

    
def reverse_hex(rstr):
    """
    This function will reverse the hex values. Ex: FEAB will be changed to ABFE
    """

    return ''.join([rstr[i:i + 2] for i in range(0, len(rstr), 2)][::
                   -1])    
    

def get_file_time(dte):
    """
    This function will convert the hex value into datetime
    """

    microseconds = int(dte, 16) / 10
    (seconds, microseconds) = divmod(microseconds, 1000000)
    (days, seconds) = divmod(seconds, 86400)
    return datetime(1601, 1, 1) + timedelta(days, seconds, microseconds)


def get_utc_from_str(stringhex):
    """
    """
    
    try:
        utc = get_file_time(stringhex)
        from_zone = tz.gettz('UTC')
        utc = utc.replace(tzinfo=from_zone)
    except Exception as e:
        utc = datetime(1970, 1, 1)
    return utc


def get_value_from_hex(file, readbytes):
    """
    """
    
    string = file.read(readbytes).hex()
    string = reverse_hex(string)
    return string


def get_value_from_hextoint(file, readbytes):
    """
    """
    
    string = file.read(readbytes).hex()
    string = reverse_hex(string)
    return int(string, 16)


def get_images(*args, **kwargs): 
    """
    """

    filenamepath = args[0]
    required_frames = args[1]
    outwrite    = kwargs.get('outwrite', False)
    applyresult = kwargs.get('applyresult', [])
    fmt         = kwargs.get('fmt', False)
    
    applyresult_flag = False
    if applyresult:
        applyresult_flag = True

    cap = cv2.VideoCapture(filenamepath)
    frame_count = 1    
    
    activity = 'Writing Image(s)' if outwrite else 'Pulling Image(s)'
    
    with Progress(activity, len(required_frames)) as pbar:
        
        for tup in range(len(required_frames)):
            frame, group = required_frames[tup]
        
            while frame_count <= frame:
                cap.grab()
                frame_count += 1
            ret, image = cap.retrieve()
       
            if outwrite:
                if fmt:
                    im = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    im.save(outwrite, fmt) 
                else:
                    if applyresult:
                        result = [] if applyresult[tup] == 0 else [eval(i) for i in applyresult[tup].split("|")]
                    else:
                        result = []
                    image = apply_model_result(frame, group, image, result, applyresult_flag)
                    outwrite.write(image)
            else:
                required_frames[tup] = (frame, group, image)
        
            pbar.update(tup + 1)
    
    if not outwrite:
        return required_frames
    

def apply_model_result(*args, **kwargs): 
    """
    """
    frame, group, image, result, applyresult_flag = args
    if applyresult_flag:
        showmessage = kwargs.get('showmessage', False)
        for coor, confi in result:
            image = cv2.rectangle(image, (coor[0], coor[1]), (coor[0] + coor[2], coor[1] + coor[3]), (255,255,0), 1)
            cv2.putText(image, str(confi), (coor[0], coor[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_8)
            
            if showmessage:
                print('Frame #{0} => Confidence(%): {1}  Co-ordinates(x,y,w,h): {2}'.format(frame,int(confi*100),coor), flush=True)
        
        cv2.putText(image, str(int(frame)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), thickness=1, lineType=cv2.LINE_8)
    return image
    

@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """
    
    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'a+')
        dest_file.write(str(datetime.now()) + '\n')
        os.dup2(dest_file.fileno(), stdchannel.fileno())
        
        yield
    except Exception as e:
        raise
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


class cv2Pseudo(): 
    """
    """
    
    def __init__(self, *args, **kwargs): 
        """
        """

        self.suppressffmpeg = kwargs.get('suppressffmpeg', False)
        self.filenamepath = args[0]
        self.dest_filename = self.filenamepath + "_ffmpeg.log"
        
    
    
    def VideoCapture(self): 
        """
        """
        if self.suppressffmpeg:
            with stdchannel_redirected(sys.stderr, self.dest_filename):
                self.cap = cv2.VideoCapture(self.filenamepath)
        else:
            self.cap = cv2.VideoCapture(self.filenamepath)
    

    def read(self): 
        """
        """
        
        if self.suppressffmpeg:
            with stdchannel_redirected(sys.stderr, self.dest_filename):
                return self.cap.read()
        else:
            return self.cap.read()


    def isOpened(self):
        """
        """

        return self.cap.isOpened()    
    
    
    def release(self):
        """
        """
        
        self.cap.release()
        

    def get(self, value):
        """
        """

        return self.cap.get(value)


class RedirectStdStreams(object):
    """
    """
    
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        
        
def sendemail(subject='Test', body='', mail_dl='adayalan1@metlife.com', sender='GCSTest@metlife.com', appconfig={'SMTP_MAIL_SRVR': 'commin.metlife.com' ,'SMTP_PORT': 25}, attachment_file=None):
    """
    Sends mail via SMTP server with the given input parameters.

    **Mandatory parameters**

    :param subject: subject of the mail

    :type subject: string

    :param body: body of the mail

    :type body: string

    :param mail_dl: mail_dl to whom the mail has to be sent

    :type mail_dl: string

    :param sender: mail id from whom the mail is sent

    :type sender: string

    **Optional parameters**

    :param attachment_file: path and filename of the attachment

    :type attachment_file: string

    """
    smtp_name = appconfig['SMTP_MAIL_SRVR']
    port_no = appconfig['SMTP_PORT']

    address_book = [mail_dl]
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ';'.join(address_book)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if attachment_file is not None:
        with open(attachment_file, "rb") as file:
            part = MIMEApplication(
                file.read(),
                Name=Path(attachment_file).name
            )
    
        part['Content-Disposition'] = 'attachment; filename="%s"' % Path(attachment_file).name
        msg.attach(part)
        
    text = msg.as_string()


    s = smtplib.SMTP(smtp_name, port_no)
    s.sendmail(sender, address_book, text)
    s.quit()


def unix_time_millis(dt):
    """
    """
    
    epoch = datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0