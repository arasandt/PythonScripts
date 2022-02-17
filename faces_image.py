import glob
import cv2
import os
from PIL import Image
from io import BytesIO
import numpy as np
import math
from mtcnn.mtcnn import MTCNN
detector = MTCNN()



def equalize(f):
    h = np.histogram(f, bins=np.arange(257))[0]
    H = np.cumsum(h) / float(np.sum(h))
    e = np.floor(H[f.flatten().astype('int')]*255.)
    return e.reshape(f.shape)

def pullfaces(filname,dpath):
    image = cv2.imread(filname)
    result_all = detector.detect_faces(image)
    print(result_all,len(result_all))
    img1 = np.copy(image)
    if result_all:
        for i,result in enumerate(result_all):
            #keypoints = result['keypoints']
            x = result['box'][0]
            y = result['box'][1]
            w = result['box'][2]
            h = result['box'][3]
            img11 = img1[y:y+h, x:x+w]
            cv2.imwrite(filname + "." + str(i) + ".cropped.png", img11)
            cv2.rectangle(image,(x, y),(x + w, y + h),(0,155,255), 2)
    else:
        print('No faced found')
    #print(bounding_box) # [x, y, width, height]
    cv2.imwrite(filname + ".withface.png", image)
    return
    
    #print(keypoints) # (x, y)

    
    
    
    

#cv2.rectangle(img1,(x,y),(w,h),(255,255,255),2)

#cv2.imwrite("ivan_drawn.jpg", img1)
    #new_x = keypoints['nose'][0] - (bounding_box[2] // 2) 
    #new_y = keypoints['nose'][1] - (bounding_box[3] // 2) 
    #print(new_x,new_y)

    #img2 = np.copy(img11)
    #myradians = math.atan2(keypoints['left_eye'][1]-keypoints['right_eye'][1], keypoints['left_eye'][0]-keypoints['right_eye'][0])
    #mydegrees = math.degrees(myradians)

    #print(img2.shape, mydegrees)
    #rows,cols,_ = img2.shape

    #M = cv2.getRotationMatrix2D((cols/2,rows/2),180 + mydegrees,1)
    #img2 = cv2.warpAffine(img2,M,(cols,rows))

#img2 = rotate_image(img2,mydegrees)
#cv2.rectangle(img2,(new_x, new_y),(new_x + bounding_box[2], new_y + bounding_box[3]),(0,155,255), 2)
#draw_angled_rec(keypoints['nose'][0],keypoints['nose'][1],bounding_box[2],bounding_box[3],mydegrees,img2)
#draw_angled_rec(new_x,new_y,bounding_box[2],bounding_box[3],-30,img2)
    #x = dpath + os.path.basename(filname) + ".rotated.png"
    #print(x)
    #cv2.imwrite(x, cv2.resize(equalize(img2), (182, 182)))
    #ima=Image.open(x)
    #with BytesIO() as f:
    #    ima.save(f, format='PNG')
        #f.seek(0)
        #ima_jpg = Image.open(f)
        #ima_jpg.save(x)
    #img3 = np.copy(image)
    #cv2.circle(img3,(keypoints['left_eye']), 2, (0,155,255), 2)
    #cv2.circle(img3,(keypoints['right_eye']), 2, (0,155,255), 2)
    #cv2.circle(img3,(keypoints['nose']), 2, (0,155,255), 2)
    #cv2.circle(img3,(keypoints['mouth_left']), 2, (0,155,255), 2)
    #cv2.circle(img3,(keypoints['mouth_right']), 2, (0,155,255), 2)
    #cv2.imwrite(filname + ".points.jpg", img3)
#cv2.namedWindow("image")
#cv2.imshow("image",image)
#cv2.waitKey(0)



fpath = 'D:\\Arasan\\Misc\\GitHub\\VideoCapture\\mtcnn\\*1.jpg'
dpath = 'D:\\Arasan\\Misc\\GitHub\\VideoCapture\\mtcnn\\'
for fil in glob.glob(fpath):
    #print(fil)
    pullfaces(fil,dpath)
