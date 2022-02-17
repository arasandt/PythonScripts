#import cv2, os
#
#filenamepath = ".\\videos\\1d5b8519ee4de9c 1035495.lnr_raw"
#temp_file = 'temp.h264'
#startposition = 17469
#
#
#
#endposition = 150521
#
#
#
#
#endbyte = b'\xff\xff\xff\xff'
#with open(filenamepath, 'rb') as binary_file:
#    with open(temp_file , 'wb') as out_file:
#        out_file.write(binary_file.read(200))
#        binary_file.seek(startposition, os.SEEK_SET)
#        out_file.write(binary_file.read()[:endposition  - startposition])
#        out_file.write(endbyte)
#        
#cap = cv2.VideoCapture(temp_file)   
#print(cap.isOpened())
##cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))     
#cap.release()


#import cv2
#import glob
#folderpath = "D:\\Arasan\\Common\\ML\\Video Analytics\\Video_Analytics_Pilot_withDB\\videos"

#cap = cv2.VideoCapture(temp_file) 
#
#import contextlib, os, sys


#@contextlib.contextmanager
#def stdchannel_redirected(stdchannel, dest_filename):
#    """
#    A context manager to temporarily redirect stdout or stderr
#
#    e.g.:
#
#
#    with stdchannel_redirected(sys.stderr, os.devnull):
#        if compiler.has_function('clock_gettime', libraries=['rt']):
#            libraries.append('rt')
#    """
#
#    try:
#        oldstdchannel = os.dup(stdchannel.fileno())
#        dest_file = open(dest_filename, 'w')
#        os.dup2(dest_file.fileno(), stdchannel.fileno())
#
#        yield
#    finally:
#        if oldstdchannel is not None:
#            os.dup2(oldstdchannel, stdchannel.fileno())
#        if dest_file is not None:
#            dest_file.close()
#
#
#import launch
#
#for file in glob.glob(folderpath + '\\*.h264'):
#    launch.main_process(filenamepath=file, process='metadata')


import pandas as pd

df = pd.read_excel('lnr_file_data_ch1.xlsx',)
df = df[['FILE_PATH','FILE_NM','FILE_ID','FRAME_RT']].reindex()
for index, row in df.iterrows():
    #filn = row['FILE_NM'].split(' ')[1].replace('.lnr','') + '.trg'
    filn = str(row['FILE_ID']) + '.trg'
    content = str(row['FILE_PATH']) + "|" + str(row['FRAME_RT'])
    with open(filn,'w') as wr:
        wr.write(content)

