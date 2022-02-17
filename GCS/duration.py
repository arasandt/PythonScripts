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

import os
import pandas as pd

from pathlib import Path
from collections import defaultdict
from datetime import datetime

from core import utils


class getDuration():
    """
    """
    
    def __init__(self, *args, **kwargs):    
        """
        Initializes all instance variable on the video
       
        """
        
        self.filenamepath = os.path.abspath(args[0])
        self.p = Path(self.filenamepath)
        self.file_size = self.p.stat().st_size
        
    
    def getstats(self, *args, **kwargs):    
        """
        """
        
        with open(self.filenamepath, 'rb') as binary_file:
            
            binary_file.seek(32, os.SEEK_SET)
            starttimehexencoded = utils.get_value_from_hex(binary_file,8)
            starttimeencoded = utils.get_file_time(starttimehexencoded)            
            starttimeinepochmill = utils.unix_time_millis(starttimeencoded)

            binary_file.seek(-4, os.SEEK_END)
            valbyte = binary_file.tell()
            val = binary_file.read(4).hex()
            temp_dict = defaultdict(dict)
            cnt = 0
            prevalbyte = self.file_size
            if val.count('f') != 8:
                valtime = -1
            else:
                while valbyte > 200 and prevalbyte > valbyte:
                    #print(cnt, valbyte, prevalbyte)
                    binary_file.seek(-8, os.SEEK_CUR)
                    binary_file.seek(-8, os.SEEK_CUR)
                    valtime = utils.get_value_from_hextoint(binary_file,8)
                    binary_file.seek(-16, os.SEEK_CUR)
                    prevalbyte = valbyte
                    valbyte = utils.get_value_from_hextoint(binary_file,8)
                    temp_dict[cnt] = {
                                    'count'         : cnt,
                                    'byteposition'  : valbyte,
                                    'frametimenano'     : valtime * 100 ,
                                   }            
                    cnt += 1
                                        
            calc_frame_count, byte_df = self._pull_bytes(binary_file)
            temp_df = pd.DataFrame.from_dict(temp_dict, 'index')
            #temp_df.drop(temp_df.tail(1).index,inplace=True)
            temp_df = temp_df.iloc[::-1]
            max_cnt = temp_df['count'].max()
            
            temp_df['count'] =  temp_df['count'].apply(lambda x: max_cnt - x + 1)
            temp_df.to_csv(self.filenamepath + '_timeraw.csv', header=True, sep=',', index=False)
            
            temp_df = pd.merge(left=temp_df, right=byte_df, left_on='byteposition', right_on='curposition')
            
            min_cnt = temp_df['count'].min()
            temp_df['count'] = temp_df['count'] - min_cnt + 1
            
            temp_df.drop(['curposition','nextposition'], axis=1, inplace=True, errors='ignore')
            temp_df['frametimenanoslice'] = temp_df['frametimenano'].diff(periods=1)
            temp_df.at[0, 'frametimenanoslice'] = temp_df['frametimenano'].iloc[0] 
            temp_df['frametimenanoslice'] = temp_df['frametimenanoslice'].apply(int)
            
            temp_df = temp_df[['count', 'byteposition', 'bytesize', 'frametimenano', 'frametimenanoslice']]
            
            temp_df['starttimeinepochmill'] = datetime.utcfromtimestamp(starttimeinepochmill/1000).strftime('%Y-%m-%d %H:%M:%S.%f')
            
            temp_df['epochendtimemilli']  = temp_df['frametimenano'].apply(lambda x : (x / 1000000) +  starttimeinepochmill)
            temp_df['epochendtimemicroformatted'] = temp_df['epochendtimemilli'].apply(lambda x: datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            
            #temp_df['timeslice'].iloc[0] = temp_df['frametimenano'].iloc[0] 
            
            print(temp_df.head())
            temp_df.to_csv(self.filenamepath + '_time.csv', header=True, sep=',', index=False)
        
        valtime = temp_df['frametimenano'].iloc[-1]
        return temp_df['count'].max(), (valtime / 1000000000)


    def _pull_bytes(self, binary_file):
        """
        """
        
        temp_dict = defaultdict(dict)
        position = 200
        binary_file.seek(position, os.SEEK_SET)
        cnt = 0
    
        while position <= self.file_size:
            val = utils.get_value_from_hextoint(binary_file, 3)
            if val == 0: break;
            binary_file.seek(val - 3, os.SEEK_CUR)
            
            temp_dict[cnt] = {
                            'curposition'   : position,
                            'bytesize'          : val,
                            'nextposition'  : position + val,
                           }            
            
            position = position + val            
            cnt += 1
        
#        temp_dict[cnt] = {
#                            'curposition'   : 0,
#                            'size'          : 0,
#                            'nextposition'  : 0,
#                           }   
        
        temp_df = pd.DataFrame.from_dict(temp_dict, 'index')
        temp_df.to_csv(self.filenamepath + '_bytes.csv', header=True, sep=',', index=False)
        return cnt, temp_df


def getFileDuration(channel_files):
    """
    """
    
    new_file_list  = []
    
    for file in channel_files:
        fileobj = getDuration(file)
        frame_count, p_duration = fileobj.getstats()
        new_file_list.append((file, frame_count, p_duration))    
        
    return new_file_list
    

if __name__ == '__main__':
    """
    """


    listofFiles = [r'D:\Arasan\Common\ML\Video Analytics\Video_Analytics_PilotV1\videos\sample.lnr',
                   r'D:\Arasan\Common\ML\Video Analytics\Pilot\videos\1d396da99c0a1c9 92142.lnr',
                   r'D:\Arasan\Common\ML\Video Analytics\Pilot\videos\1d4add6e1d9139c 173602.lnr',
                  ]

    listofFiles = [ #r'D:\Arasan\Common\ML\Video Analytics\Video_Analytics_PilotV1\videos\1d5ef74fe405a88 1689751.lnr',
                    #r'D:\Arasan\Common\ML\Video Analytics\Video_Analytics_PilotV1\videos\sample.lnr'
                    r'D:\Arasan\Common\ML\Video Analytics\Video_Analytics_PilotV1\videos\1d5f09115aff29c 2884725.lnr',
                  ]
    
    new_file_list = getFileDuration(listofFiles)
    print(new_file_list)
    