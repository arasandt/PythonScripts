import cv2
#import glob
#images = glob.glob("D:\\Arasan\\Misc\\GitHub\\Others\\input\\*.jpg")
# =============================================================================
# img = cv2.imread("D:\\Arasan\\Misc\\GitHub\\Others\\input\\galaxy.jpg",0) #color=1, grayscale=0 and -1 --> not sure
# resized_image=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
# cv2.imshow('galaxy', resized_image)
# cv2.imwrite('galaxy_resized.jpg', resized_image)
# cv2.waitKey(0) # 0 will mean hit any key. here 2000 means seconds
# cv2.destroyAllWindows()
# 
# 
# =============================================================================


face_cascade =  cv2.CascadeClassifier("D:\\Arasan\\Misc\\GitHub\\Others\\input\\Files\\haarcascade_frontalface_default.xml")
img=cv2.imread("D:\\Arasan\\Misc\\GitHub\\Others\\input\\Files\\photo.jpg")
#img=cv2.imread("D:\\Arasan\\Misc\\GitHub\\Others\\input\\Files\\news.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img,
                                    scaleFactor=1.1,
                                    minNeighbors=5)
for x, y, w, h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

print(type(faces))
print(faces)

resized=cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))

cv2.imshow('grey',resized)
cv2.waitKey(0)
cv2.destroyAllWindows()