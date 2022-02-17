# =============================================================================
# 
# 
# import cv2
# import numpy as np
# 
# cap = cv2.VideoCapture("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\car.mp4")
# #ret, current_frame = cap.read()
# #previous_frame = current_frame
# c = 0
# while True:
#     ret, current_frame = cap.read()
#     if ret == False:
#         break
#     if c == 0:
#         previous_frame = current_frame.copy()
#         c += 1
#         continue
#     current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
#     previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    
#     #current_frame_gray = current_frame
#     #previous_frame_gray = previous_frame
#     #current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0) 
#     frame_diff = cv2.absdiff(previous_frame_gray,current_frame_gray)
#     #gray = cv2.bilateralFilter(current_frame_gray, 11, 17, 17)
#     #edged = cv2.Canny(gray, 30, 200)
#     #frame_diff = cv2.subtract(current_frame_gray,previous_frame_gray)
#     thresh_frame = cv2.threshold(current_frame_gray, 10, 255, cv2.THRESH_BINARY)[1] 
#     thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
#     # Finding contour of moving object 
#     (cnts, _) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#     #for contour in cnts: 
#     #    if cv2.contourArea(contour) < 10000: 
#     #        continue
#         #motion = 1
#     cv2.drawContours(thresh_frame, cnts, -1, (0,255,0), 1)
#         #(x, y, w, h) = cv2.boundingRect(contour) 
#         # making green rectangle arround the moving object 
#         #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)     
# 
#     #ret, thresh = cv2.threshold(frame_diff,127,255,0)
#     #contours, hierarchy = cv2.findContours(thresh_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     #cv2.drawContours(frame_diff, contours, -1, (0,255,0), 1)
#     #x = np.allclose(previous_frame_gray,current_frame_gray, rtol=1e-05, atol=1e-08)
#     
#     #if x: 
#     #    print('Frame {0} is same'.format(c))
#     #cv2.imshow('frame diff ',frame_diff)      
#     cv2.imshow('actual frame', current_frame_gray)      
#     cv2.imshow('thresh frame', thresh_frame)      
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# 
#     previous_frame = current_frame.copy()
#     #ret, current_frame = cap.read()
#     c += 1
# cap.release()
# cv2.destroyAllWindows()
# 
# =============================================================================

# Pyhton program to implement  
# WebCam Motion Detector 
  
# importing OpenCV, time and Pandas library 
import cv2, time, pandas
import numpy as np 
# importing datetime class from datetime library 
from datetime import datetime 
  
# Assigning our static_back to None 
static_back = None
  
# List when any moving object appear 
motion_list = [ None, None ] 
  
# Time of movement 
captime = [] 
  
# Initializing DataFrame, one column is start  
# time and other column is end time 
df = pandas.DataFrame(columns = ["Start", "End"]) 
  
# Capturing video 
video = cv2.VideoCapture("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\no.mp4")
frame_width = int(video.get(3))
frame_height = int(video.get(4))
frame_rate = int(video.get(cv2.CAP_PROP_FPS))
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter("outpy.avi",cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width * 2,frame_height))
c= 0  
saved = 0
# Infinite while loop to treat stack of image as video 
while True: 
    # Reading frame(image) from video 
    check, frame = video.read() 
    if check == False: break
    # Initializing motion = 0(no motion) 
    motion = 0
  
    # Converting color image to gray_scale image 
    grayx = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Converting gray scale image to GaussianBlur  
    # so that change can be find easily 
    gray = cv2.GaussianBlur(grayx, (21, 21), 0) 
  
    # In first iteration we assign the value  
    # of static_back to our first frame 
    if static_back is None: 
        static_back = gray 
        continue

    if c % 10 == 0 and saved == 0: 
        saved = 1
        save_static_back = gray 
    if c % 20 == 0 and saved == 1:
        static_back = save_static_back
        saved = 0
  
    # Difference between static background  
    # and current frame(which is GaussianBlur) 
    diff_frame = cv2.absdiff(static_back, gray) 
  
    # If change in between static background and 
    # current frame is greater than 30 it will show white color(255) 
    thresh_frame = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)[1] 
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
  
    # Finding contour of moving object 
    (cnts, _) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    #print(cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) )
  
    for contour in cnts: 
        if cv2.contourArea(contour) < 10000: 
            continue
        motion = 1
  
        (x, y, w, h) = cv2.boundingRect(contour) 
        # making green rectangle arround the moving object 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 
  
    # Appending status of motion 
    motion_list.append(motion) 
  
    motion_list = motion_list[-2:] 
  
    # Appending Start time of motion 
    if motion_list[-1] == 1 and motion_list[-2] == 0: 
        captime.append(datetime.now()) 
  
    # Appending End time of motion 
    if motion_list[-1] == 0 and motion_list[-2] == 1: 
        captime.append(datetime.now()) 
  
    # Displaying image in gray_scale 
    #cv2.namedWindow('Gray Frame',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Gray Frame', frame_width//2,frame_height//2)
    #cv2.imshow("Gray Frame", gray) 
  
    # Displaying the difference in currentframe to 
    # the staticframe(very first_frame) 
    #cv2.namedWindow('Difference Frame',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Difference Frame', frame_width//2,frame_height//2)
    #cv2.imshow("Difference Frame", diff_frame) 
  
    # Displaying the black and white image in which if 
    # intencity difference greater than 30 it will appear white 
    #cv2.namedWindow('Threshold Frame',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Threshold Frame', frame_width//2,frame_height//2)
    #cv2.imshow("Threshold Frame", thresh_frame) 
    vis = np.concatenate((grayx, thresh_frame), axis=1)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    out.write(vis)
    cv2.imshow("Color Frame", vis) 
    # Displaying color frame with contour of motion of object 
    #cv2.imshow("Color Frame", frame) 
    c += 1
    #time.sleep(0.02)
    print('{0} --> {1} --> {2}'.format(c,length,np.mean(thresh_frame)))
    key = cv2.waitKey(1) 
    # if q entered whole process will stop 
    if key == ord('q'): 
        # if something is movingthen it append the end time of movement 
        if motion == 1: 
            captime.append(datetime.now()) 
        break
  
# Appending time of motion in DataFrame 
for i in range(0, len(captime), 2): 
    df = df.append({"Start":captime[i], "End":captime[i + 1]}, ignore_index = True) 
  
# Creating a csv file in which time of movements will be saved 
df.to_csv("Time_of_movements.csv") 
  
video.release() 
out.release()
# Destroying all the windows 
cv2.destroyAllWindows() 




# =============================================================================
# import numpy as np
# import cv2
# 
# sdThresh = 5
# font = cv2.FONT_HERSHEY_SIMPLEX
# #TODO: Face Detection 1
# 
# def distMap(frame1, frame2):
#     """outputs pythagorean distance between two frames"""
#     frame1_32 = np.float32(frame1)
#     frame2_32 = np.float32(frame2)
#     diff32 = frame1_32 - frame2_32
#     norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
#     dist = np.uint8(norm32*255)
#     return dist
# 
# cv2.namedWindow('frame')
# cv2.namedWindow('dist')
# 
# #capture video stream from camera source. 0 refers to first camera, 1 referes to 2nd and so on.
# cap = cv2.VideoCapture("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\car.mp4")
# 
# _, frame1 = cap.read()
# _, frame2 = cap.read()
# 
# facecount = 0
# c = 0
# while(True):
#     ret , frame3 = cap.read()
#     if ret == False or c == 1000: break
#     rows, cols, _ = np.shape(frame3)
#     cv2.imshow('dist', frame3)
#     dist = distMap(frame1, frame3)
# 
#     frame1 = frame2
#     frame2 = frame3
# 
#     # apply Gaussian smoothing
#     mod = cv2.GaussianBlur(dist, (9,9), 0)
# 
#     # apply thresholding
#     _, thresh = cv2.threshold(mod, 100, 255, 0)
# 
#     # calculate st dev test
#     _, stDev = cv2.meanStdDev(mod)
#     
#     cv2.imshow('dist', mod)
#     cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
#     c += 1
#     if stDev > sdThresh:
#             print('Frame {0} --> Motion detected.. Do something!!!'.format(c))
#             #TODO: Face Detection 2
# 
#     cv2.imshow('frame', frame2)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     
#     
# cap.release()
# cv2.destroyAllWindows()
# 
# =============================================================================


# =============================================================================
# import cv2 as cv
# from datetime import datetime
# import time
# #import numpy as np
# 
# class MotionDetectorInstantaneous():
#     
#     def onChange(self, val): #callback when the user change the detection threshold
#         self.threshold = val
#     
#     def __init__(self,threshold=1, doRecord=True, showWindows=True):
#         self.writer = None
#         #self.font = None
#         self.doRecord = doRecord #Either or not record the moving object
#         self.show = showWindows #Either or not show the 2 windows
#         self.frame = None
#     
#         self.capture= cv.VideoCapture("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\car.mp4")
#         _, self.frame = self.capture.read() #Take a frame to init recorder
#         self.width = int(self.capture.get(3))
#         self.height = int(self.capture.get(4))
# 
#         if doRecord:
#             self.initRecorder()
#         
#         #self.frame1gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t-1
#         self.frame1gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY) 
#         #cv.CvtColor(self.frame, self.frame1gray, cv.CV_RGB2GRAY)
#         
#         #Will hold the thresholded result
#         #self.res = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U)
#         #self.frame2gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t
#         self.frame2gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY) 
#         self.nb_pixels = self.width * self.height
#         self.threshold = threshold
#         self.isRecording = False
#         self.trigger_time = 0 #Hold timestamp of the last detection
#         
#         if showWindows:
#             #cv.NamedWindow("Image")
#             cv.imshow('frame', self.frame)
#             #cv.CreateTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)
#         
#     def initRecorder(self): #Create the recorder
#         #self.frame_width = int(self.capture.get(3))
#         #self.frame_height = int(self.capture.get(4))
#         self.frame_rate = int(self.capture.get(cv.CAP_PROP_FPS))
#         #codec = cv.CV_FOURCC('M', 'J', 'P', 'G') #('W', 'M', 'V', '2')
#         #self.writer=cv.CreateVideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv", codec, 5, cv.GetSize(self.frame), 1)
#         self.out = cv.VideoWriter("outpy.avi",cv.VideoWriter_fourcc(*'XVID'), self.frame_rate, (self.width,self.height))
#         #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
#         #self.font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8) #Creates a font
# 
#     def run(self):
#         started = time.time()
#         while True:
#             ret, curframe = self.capture.read()
#             
#             if ret == False: 
#                 break
#             
#             instant = time.time() #Get timestamp of the frame
#             
#             self.processImage(curframe) #Process the image
#             
#             if not self.isRecording:
#                 if self.somethingHasMoved():
#                     self.trigger_time = instant #Update the trigger_time
#                     if (instant > started + 5): #Wait 5 second after the webcam start for luminosity adjusting etc..
#                         print (datetime.now().strftime("%b %d, %H:%M:%S"), "Something is moving !")
#                         if self.doRecord: #set isRecording=True only if we record a video
#                             self.isRecording = True
#             else:
#                 if instant >= self.trigger_time + 10: #Record during 10 seconds
#                     print(datetime.now().strftime("%b %d, %H:%M:%S"), "Stop recording")
#                     self.isRecording = False
#                 else:
#                     #cv.putText(curframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),cv.FONT_HERSHEY_SIMPLEX, 0) #Put date on the frame
#                     cv.putText(curframe, datetime.now().strftime("%b %d, %H:%M:%S"), (25, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
#                     #cv.WriteFrame(self.writer, curframe) #Write the frame
#                     self.out.write(curframe)
#             
#             if self.show:
#                 #print(type(curframe))
#                 #print(curframe)
#                 cv.imshow('Image', curframe)
#                 cv.imshow('Res', self.res)
#                 #cv.ShowImage("Image", curframe)
#                 #cv.ShowImage("Res", self.res)
#                 
#             #cv.Copy(self.frame2gray, self.frame1gray)
#             self.frame1gray = self.frame2gray.copy()
#             if cv.waitKey(1) & 0xFF == ord('q'):
#                 break        
#     
#     def processImage(self, frame):
#         #cv.CvtColor(frame, self.frame2gray, cv.CV_RGB2GRAY)
#         self.frame2gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
#         
#         #Absdiff to get the difference between to the frames
#         #cv.AbsDiff(self.frame1gray, self.frame2gray, self.res)
#         self.res = cv.absdiff(self.frame1gray, self.frame2gray) 
#         #Remove the noise and do the threshold
#         self.res = cv.GaussianBlur(self.res, (9,9), 0)
#         #cv.Smooth(self.res, self.res, cv.CV_BLUR, 5,5)
#         #kernel = cv.getStructuringElement(shape=cv.MORPH_OPEN, ksize=(3,3))
#         #self.res = cv.morphologyEx(self.res,cv.MORPH_OPEN, kernel) #,iterations = 3)
#         #self.res = cv.morphologyEx(self.res,cv.MORPH_OPEN, kernel) #,iterations = 3)
#         #_, self.res = cv.threshold(self.res, 10, 255, cv.THRESH_BINARY_INV)
# 
#     def somethingHasMoved(self):
#         nb=0 #Will hold the number of black pixels
#         min_threshold = (self.nb_pixels/100) * self.threshold #Number of pixels for current threshold
#         nb = self.nb_pixels - cv.countNonZero(self.res)
#         if (nb) > min_threshold:
#            return True
#         else:
#            return False
#         
# if __name__=="__main__":
#     detect = MotionDetectorInstantaneous(doRecord=True)
#     detect.run()
# =============================================================================
