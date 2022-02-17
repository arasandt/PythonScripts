import cv2
import numpy as np
import os
 
# Create a VideoCapture object
#cap = cv2.VideoCapture("D:\\Arasan\\Misc\\GitHub\\FaceNetM\\akshay_mov.mp4")
cap = cv2.VideoCapture("D:\\Arasan\\Misc\\GitHub\\BigDecisions\\Data\\input\\Met1 MDF Outside.lnr")
out_folder = './images/' 

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
#frame_width = int(cap.get(3))
#frame_height = int(cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter("D:\\Arasan\\Misc\\GitHub\\VideoCapture\\Juice Bar 20fps.lnr.output.mp4",cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width,frame_height))
#out = cv2.VideoWriter('D:\\Arasan\\Misc\\GitHub\\FaceNetM\\outpy.mp4',cv2.VideoWriter_fourcc('M','P','V','4'), 20.0, (640,480))
#print(out) 
i = 1
while(True):
  ret, frame = cap.read()
  if ret == True: 
     
    # Write the frame into the file 'output.avi'
    try:
        #out.write(frame)
        
        cv2.imwrite(os.path.join(out_folder,str(i) + '.jpg'), frame)
        i += 1
    except Exception as e:
        print(e)
        
    
    # Display the resulting frame    
    #cv2.imshow('frame',frame)
 
    # Press Q on keyboard to stop recording
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #  break
 
  # Break the loop
  else:
    break 
 
# When everything done, release the video capture and video write objects
cap.release()
#out.release()
 
# Closes all the frames
#v2.destroyAllWindows() 