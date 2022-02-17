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

files = files[:1]

def reversehex(x):
    return ''.join([x[i:i+2] for i in range(0, len(x), 2)][::-1])

def getFiletime(dt):
        microseconds = int(dt, 16) / 10
        seconds, microseconds = divmod(microseconds, 1000000)
        days, seconds = divmod(seconds, 86400)
        return datetime(1601, 1, 1) + timedelta(days, seconds, microseconds)

def filebytes(f,size=1):
    i = 0
    while True:
        f.seek(-i-1,os.SEEK_END)
        i += 1
        yield f.read(size).hex()

def printmetadata(fil):
    with open(fil, "rb") as binary_file:
        
        binary_file.seek(32, 0)
        utc = binary_file.read(8).hex()
        utc = getFiletime(reversehex(utc))
        from_zone = tz.gettz('UTC')
        to_zone = tz.gettz('America/New_York')    
        utc = utc.replace(tzinfo=from_zone)
        est = utc.astimezone(to_zone)
        
        
        binary_file.seek(164, 0)
        width = binary_file.read(2).hex()
        width = int(reversehex(width),16)
        
    
        binary_file.seek(168, 0)
        height = binary_file.read(2).hex()
        height = int(reversehex(height),16)
        
        
        from collections import deque
        L = deque(maxlen=16)
        filgen = filebytes(binary_file)
        
        while True: # at some point convert 
        #for i in range(30):
            data = next(filgen)
            L.append(data)
            if '0000000000c80000' in ''.join(list(L)):
                #print(list(L))
                break
        binary_file.seek(-11, 1)
        coded_frames = binary_file.read(2).hex()
        coded_frames = int(reversehex(coded_frames),16)
        #print(coded_frames)
        binary_file.seek(2, 1)
        frames = binary_file.read(8).hex()
        frames = int(reversehex(frames),16)
        #print(frames)
        
        frame_rate = round(coded_frames*1000*10000/frames)
        
        est_endtime = est + timedelta(seconds=round(coded_frames/frame_rate)) 
        
        print('File : {0}'.format(os.path.basename(fil)))
        print('Start Time : {0}'.format(est.strftime('%m/%d/%Y %I:%M:%S %p %Z')))
        print('End Time   : {0}'.format(est_endtime.strftime('%m/%d/%Y %I:%M:%S %p %Z')))
        print('Video Size (Width)  : {0}'.format(width))
        print('Video Size (Height) : {0}'.format(height))
        print('Total Frames : {0}'.format(round(coded_frames)))
        print('Frame Rate (fps) : {0}'.format(frame_rate))
        
        
        
        

    #print(list(L))
#    binary_file.seek(-1024 * 1024, os.SEEK_END)
#    print(binary_file.read().hex())
        
#print (getFiletime('01d4add82ff95e80'))
#print (getFiletime('01d4c9fb047f1a80'))
for fil in files:
    printmetadata(fil)
    print()


# =============================================================================
# fil = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\testing_forendtimestamp.lnr"
# fil1 = "D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Lenel_video_LNR_Format.lnr"
# 
# def intconv(x):
#     #print(x)
#     x = x.hex()#[::-1]
#     x = ''.join([x[i:i+2] for i in range(0, len(x), 2)][::-1])
#     return (int("0x" + x,16))
# 
#     
# all_data = []
# all_data1 = []
# with open(fil, "rb") as binary_file:
#     binary_file.seek(45, 1)
#     data = intconv(binary_file.read(2))
#     #print(data)
#     #data = data.hex()#[::-1]
#     #data = ''.join([data[i:i+2] for i in range(0, len(data), 2)][::-1])
#     binary_file.seek(10, 1)
#     i = 0
#     savedata = 0
#     while True:
#         try:
#             data = intconv(binary_file.read(3))
#             binary_file.seek(5, 1)
#             data1 = intconv(binary_file.read(4))
#             data1 = (str(data1)[:-6],str(data1)[-6:])
#             #print(savedata,data)
#             if savedata > data:
#                 break
#             else:
#                 all_data.append(data)
#                 all_data1.append(data1)
#             savedata = data
#             binary_file.seek(4, 1)
#             #if i == 40:
#             #    break
#             i += 1
#         except Exception:
#             break
#     
# #print(all_data)
# 
# i = 0
# all_data = all_data[:10]
# #all_data1 = all_data1[-10:]
# #print(len(all_data))
# #print(all_data)
# #print(all_data1)
# sumdata1 = sum([int(j) for i,j in all_data1])
# #print(sumdata1)
# with open(fil1, "rb") as binary_file:
#     for i in all_data:
#         binary_file.seek(i, 0)
#         data = intconv(binary_file.read(2))
#         print(data)
#     
# =============================================================================
