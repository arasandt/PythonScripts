from datetime import datetime, timedelta
from dateutil import tz
import os

# =============================================================================
# data = ['01d4add87cdd3500','1d4750c78ba7c43','1d4750d741c67b6','1d4750e64060517','1d4750f3b32ef0d']
# 
# 
# def getFiletime(dt):
#         microseconds = int(dt, 16) / 10
#         seconds, microseconds = divmod(microseconds, 1000000)
#         days, seconds = divmod(seconds, 86400)
#         return datetime(1601, 1, 1) + timedelta(days, seconds, microseconds)
# 
# 
# from_zone = tz.gettz('UTC')
# to_zone = tz.gettz('America/New_York')    
# filetime = getFiletime(data[1])
# utc = filetime
# utc = utc.replace(tzinfo=from_zone)
# est = utc.astimezone(to_zone)
# print(est.strftime('%m/%d/%Y %I:%M:%S %p %Z'))
# =============================================================================


#fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1MDFInside.lnr"
#fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1MDFOutside.lnr"
#fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met2SW.lnr"
#fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1Juicebar.lnr"
#fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1MDFEntrance.lnr"
#fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Lenel_video_LNR_Format.lnr"
#fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Juice Bar 20fps.lnr"

files = ["D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1MDFInside.lnr",
         "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1MDFOutside.lnr",
         "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met2SW.lnr",
         "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1Juicebar.lnr",
         "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Met1MDFEntrance.lnr",
         "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Lenel_video_LNR_Format.lnr",
         "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Juice Bar 20fps.lnr",
         "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\1d4add6e1d9139c 173602.lnr"
         ]

#files = files[-2:-1]

def reversehex(x):
    return ''.join([x[i:i+2] for i in range(0, len(x), 2)][::-1])

def getFiletime(dt):
        microseconds = int(dt, 16) / 10
        seconds, microseconds = divmod(microseconds, 1000000)
        days, seconds = divmod(seconds, 86400)
        return datetime(1601, 1, 1) + timedelta(days, seconds, microseconds)


def printmetadata(fil):
    with open(fil, "rb") as binary_file:
        
        binary_file.seek(32, os.SEEK_SET)
        utc = binary_file.read(8).hex()
        utc = getFiletime(reversehex(utc))
        from_zone = tz.gettz('UTC')
        to_zone = tz.gettz('America/New_York')    
        utc = utc.replace(tzinfo=from_zone)
        est = utc.astimezone(to_zone)
        
        
        binary_file.seek(164, os.SEEK_SET)
        width = binary_file.read(2).hex()
        width = int(reversehex(width),16)
        
    
        binary_file.seek(168, os.SEEK_SET)
        height = binary_file.read(2).hex()
        height = int(reversehex(height),16)
        
        
        
        
        binary_file.seek(-24,os.SEEK_END)
        lfadd = binary_file.read(4).hex()
        lfadd = int(reversehex(lfadd),16)
        #print(lfadd)
        binary_file.seek(lfadd,os.SEEK_SET)
        lfsize = binary_file.read(2).hex()
        lfsize = int(reversehex(lfsize),16)
        #print(lfsize)
        lfendadd = lfadd + lfsize
        #print(lfendadd)
        binary_file.seek(lfendadd, os.SEEK_SET)
        from collections import deque
        L = deque(maxlen=8)
# =============================================================================
#         while True:
#             L.append(binary_file.read(1).hex())
#             if '0000c800000000' in ''.join(list(L)):
#                 #print(list(L))
#                 break
#             #print(list(L))
#         
# =============================================================================
        while '0000c800000000' not in ''.join(list(L)):
            L.append(binary_file.read(1).hex())
        
        binary_file.seek(-17,os.SEEK_CUR)
        coded_frames = binary_file.read(2).hex()
        coded_frames = int(reversehex(coded_frames),16)        
        #print(coded_frames)
        
        binary_file.seek(2, 1)
        frames = binary_file.read(8).hex()
        frames = int(reversehex(frames),16)
        #print(frames)

        frame_rate = round(coded_frames*10000000/frames)
        
        est_endtime = est + timedelta(seconds=round(coded_frames/frame_rate)) 
        
        print('File                   : {0}'.format(os.path.basename(fil)))
        print('Total Frames           : {0}'.format(round(coded_frames)))
        print('Size (Width x Height)  : {0} x {1}'.format(width,height))
        print('Start Time             : {0}'.format(est.strftime('%m/%d/%Y %I:%M:%S %p %Z')))
        print('End Time               : {0}'.format(est_endtime.strftime('%m/%d/%Y %I:%M:%S %p %Z')))
        print('Frame Rate (fps)       : {0}'.format(frame_rate))
        
        
        
        
        
        
# =============================================================================
#         while True: # at some point convert 
#         #for i in range(30):
#             data = next(filgen)
#             L.append(data)
#             if '0000000000c80000' in ''.join(list(L)):
#                 #print(list(L))
#                 break
#         binary_file.seek(-11, 1)
#         coded_frames = binary_file.read(2).hex()
#         coded_frames = int(reversehex(coded_frames),16)
#         #print(coded_frames)
#         binary_file.seek(2, 1)
#         frames = binary_file.read(8).hex()
#         frames = int(reversehex(frames),16)
#         #print(frames)
#         
#         frame_rate = round(coded_frames*1000*10000/frames)
#         
#         est_endtime = est + timedelta(seconds=round(coded_frames/frame_rate)) 
#         
#         print('File : {0}'.format(os.path.basename(fil)))
#         print('Start Time : {0}'.format(est.strftime('%m/%d/%Y %I:%M:%S %p %Z')))
#         print('End Time   : {0}'.format(est_endtime.strftime('%m/%d/%Y %I:%M:%S %p %Z')))
#         print('Video Size (Width)  : {0}'.format(width))
#         print('Video Size (Height) : {0}'.format(height))
#         print('Total Frames : {0}'.format(round(coded_frames)))
#         print('Frame Rate (fps) : {0}'.format(frame_rate))
#         
# =============================================================================
        
        
        

for fil in files:
    printmetadata(fil)
    print()

