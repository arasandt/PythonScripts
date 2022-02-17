import cv2
#from PIL import Image
import numpy as np
import math
filname = "test1.jpg.rotated.jpg"
image = cv2.imread(filname)

def normalize(f):
    lmin = float(f.min())
    lmax = float(f.max())
    return np.floor((f-lmin)/(lmax-lmin)*255.)


def equalize(f):
    h = np.histogram(f, bins=np.arange(256))[0]
    #print(h)
    H = np.cumsum(h) / float(np.sum(h))
    #print(H.shape)
    e = np.floor(H[f.flatten().astype('int')]*254.)
    #e1 = f.flatten().astype('int')
    #print(e1.shape)
    #print(e1) 
    #print([x for x in e1 if x >= 254])
    #e2 = H[e1] * 255.0
    #e = np.floor(e2)
    return e.reshape(f.shape)


cv2.imwrite(filname + ".normalize.jpg", normalize(image))
cv2.imwrite(filname + ".equalize.jpg", equalize(image))