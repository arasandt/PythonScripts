import numpy as np
from scipy import misc
import cv2

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size: #only when embedded image size is less than the actual image size
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image


def flip(image, random_flip):
    if random_flip: # and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


img = misc.imread("D:\\Arasan\\Misc\\GitHub\ML\\tf_hub\\examples\\image_retraining\\test_images\\img1.jpg")
img = crop(img, False, 160)

cv2.imwrite("a.cropped.jpg", img)

img = misc.imread("D:\\Arasan\\Misc\\GitHub\ML\\tf_hub\\examples\\image_retraining\\test_images\\img1.jpg")
img = flip(img,True)

cv2.imwrite("a.flipped.jpg", img)
