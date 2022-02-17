#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains all functions which would manipulate on a single video
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, re

from collections import deque
from dateutil import tz
from datetime import datetime, timedelta


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


def get_metadata(*args, **kwargs):
    """
    This function will pull other metadata information from LNR video
    """
    
    hex32changed, partialfile = False, False
    
    with open(args[0], 'r+b') as binary_file:
#        binary_file.seek(8 * 4, os.SEEK_SET)
#        val = binary_file.read(1).hex()
#        if val == '01':
#            update = b'\x00'
#            binary_file.write(update)
#            print("32'nd byte in input video file updated to 00 ...")
#            hex32changed = True
        binary_file.seek(-4, os.SEEK_END)
        val = binary_file.read(4).hex()
        if val.count('f') != 8:
            #raise Exception('Invalid EOF in input video file. Got {0} instead of ffffffff'.format(val))
            print('Invalid EOF in input video file. Got {0} instead of ffffffff. Using default values.'.format(val))
            utchex = os.path.basename(args[0]).split(" ")[0]
            utc = get_file_time(utchex)
            from_zone = tz.gettz('UTC')
            utc = utc.replace(tzinfo=from_zone)
            partialfile = True
            return (
                0,      #height
                0,      #width
                0,      #duration
                10,     #frame_rate
                utchex,
                utc,
                0,      #utc_endtime
                999999, #coded_frames
                "",     #cameraname
                hex32changed,
                partialfile, 
                )

    with open(args[0], 'rb') as binary_file:

        binary_file.seek(32, os.SEEK_SET)
        utchex = binary_file.read(8).hex()
        utchex = reverse_hex(utchex)
        utc = get_file_time(utchex)
        from_zone = tz.gettz('UTC')
        utc = utc.replace(tzinfo=from_zone)

        binary_file.seek(164, os.SEEK_SET)
        width = binary_file.read(2).hex()
        width = int(reverse_hex(width), 16)

        binary_file.seek(168, os.SEEK_SET)
        height = binary_file.read(2).hex()
        height = int(reverse_hex(height), 16)
        
        binary_file.seek(-1, os.SEEK_END)
        last_byte = binary_file.tell()
        
        binary_file.seek(-24, os.SEEK_END)
        lfadd = binary_file.read(4).hex()
        lfadd = int(reverse_hex(lfadd), 16)
        binary_file.seek(lfadd, os.SEEK_SET)
        lfsize = binary_file.read(4).hex()
        lfsize = int(reverse_hex(lfsize), 16)
        lfendadd = lfadd + lfsize
        binary_file.seek(lfendadd, os.SEEK_SET)
        findstr = deque(maxlen=8)
        cname = list()

        while '0000c800000000' not in ''.join(list(findstr)):
            tmp = binary_file.read(1)
            findstr.append(tmp.hex())
            cname.append(chr(int(tmp.hex(), 16)))
        
        binary_file.seek(-17, os.SEEK_CUR)
        coded_frames = binary_file.read(4).hex()
        coded_frames = int(reverse_hex(coded_frames), 16)
        
        while coded_frames == 1:
            binary_file.seek(-24, os.SEEK_CUR)
            coded_frames = binary_file.read(4).hex()
            coded_frames = int(reverse_hex(coded_frames), 16)
            
        frames = binary_file.read(8).hex()
        frames = int(reverse_hex(frames), 16)
        
        cur_pos = binary_file.tell()
        coded_frames_new = (last_byte - cur_pos - 7) // 16

        frame_rate = round(coded_frames * 10000000 / frames)
        
        if frame_rate == 0:
            frame_rate = round(coded_frames_new * 10000000 / frames)
            coded_frames = coded_frames_new
        #print(coded_frames_new * 10000000 / frames)
        #0/0
        utc_endtime = utc + timedelta(seconds=round(coded_frames
                / frame_rate))
        duration = (utc_endtime - utc).total_seconds()
        
        cameraname = ''.join(re.sub('[^A-Za-z0-9-/:]+', '',
                     ''.join(cname)))

        (_, delim, end) = cameraname.partition('US')
        (cameraname, _, _) = (delim + end).partition('/')
        cameraname = re.sub('([0-9]+)$', '', cameraname)
        
        return (
            height,
            width,
            int(duration),
            frame_rate,
            utchex,
            utc,
            utc_endtime,
            coded_frames,
            cameraname,
            hex32changed,
            partialfile,
            )


