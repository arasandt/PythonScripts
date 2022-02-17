#import cv2
#from datetime import datetime, timedelta

## cap = cv2.VideoCapture("C:\Users\708743\Desktop\Met1MDFOutside.lnr.output.avi")
#cap = cv2.VideoCapture("Met1 Juice Bar.lnr.XVID.avi")
#fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS" number of frame in 1 second 10
#frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frame count in a video 80
#print(frame_count)
#duration = frame_count / fps  # 80/10 =8.0
#print(duration)
#
## user input timing
#def srt_file():
#    file = open("video.srt", "w+")
#    orig_date_L = "00:00:00"
#    start_time = datetime(2019, 11, 22, 00, 00, 00)
#    for i in range(int(duration)):
#        file.write(str(i) + '\n')
#        dt4 = datetime.strptime(str(orig_date_L), "%H:%M:%S")
#        dt4 = dt4 + timedelta(seconds=i)
#        dt4 = datetime.strptime(str(dt4), "%Y-%m-%d %H:%M:%S").time()
#        file.write(str(dt4)+',000'+"-->"+str(dt4)+',999' + '\n')
#        dt5 = datetime.strptime(str(start_time), "%Y-%m-%d %H:%M:%S")
#        dt5 = dt5 + timedelta(seconds=i)
#        dt5 = datetime.strptime(str(dt5), "%Y-%m-%d %H:%M:%S")
#        file.write(str(dt5) + '\n')
#        file.write('\n')
#    return file
#
#
#def main():
#    srt_file()
#

if __name__ == "__main__":
    #main()
    import Process
    
    # ctx, filepath, process, modelfolder, framenum
    
    x = Process.main_process(filepath= 'videos\Met1 Front to Parking.lnr', #"1d59a7544a38a01 397902.lnr", 
    #x = Process.main_process(filepath= '1d396da99c0a1c9 92142.lnr',
    #x = Process.main_process(filepath="1d59a7544a38a01 397902.lnr", 
                             modelfolder="models\yolov3-spp")
    failed_ones = {i:j for i,j in x.items() if not j[0]}
    print(failed_ones)
    
    
#    f = "C:\\Users\\128537\\Downloads\\ffmpeg-20191108-e700038-win64-static\\ffmpeg-20191108-e700038-win64-static\\bin\\1d59a7544a38a01 397902.lnr"
#    import os
#    with open(f, 'r+b') as binary_file:
#        binary_file.seek(8 * 4, os.SEEK_SET)
#        x = binary_file.read(8).hex()
#        print('Before: ' + str(x))
#        #y = b'\x00\x00\x2c\x01\x00\x00\x00\x00'
#        #y = b'\x00\x00\x00\x00\x00\x00\x00\x00'
#        #y = b'\x00\xa6\x7b\xfe\x0d\xce\xd4\x01'
#        #y = b'\x9c\x13\xd9\xe1\xd6\xad\xd4\x01'
#        #y = b'\x01\x8a\xa3\x44\x75\x9a\xd5\x01'
#        #y = b'\x00\x8a\xa3\x44\x75\x9a\xd5\x01'
#        y = b'\x00'
#        binary_file.seek(-8, os.SEEK_CUR)
#        binary_file.write(y)
#        binary_file.seek(-1, os.SEEK_CUR)
#        x = binary_file.read(8).hex()
#        print('After : ' + str(x))
        

