import sys
import cv2
import numpy
from scipy.ndimage import label

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255/ncc)
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl


img = cv2.imread(sys.argv[1])

# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
_, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, numpy.ones((3, 3), dtype=int))

result = segment_on_dt(img, img_bin)
cv2.imwrite(sys.argv[2], result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
cv2.imwrite(sys.argv[3], img)

#import cv2
#import numpy as np

#img = cv2.imread('coins.jpg')
#cv2.imshow('original',img)
#gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#ret, thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#cv2.imshow('threshold',thresh)

## noise removal
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#cv2.imshow('opening',opening)

## sure background area
#sure_bg = cv2.dilate(opening,kernel,iterations=3)
#cv2.imshow('sure_bg',sure_bg)

## Finding sure foreground area
#dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
#cv2.imshow('dist_transform',dist_transform)
#ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#cv2.imshow('sure_fg',sure_fg)

## Finding unknown region
#sure_fg = np.uint8(sure_fg)
#unknown = cv2.subtract(sure_bg,sure_fg)

## Marker labelling
#ret, markers = cv2.connectedComponents(sure_fg)

## Add one to all labels so that sure background is not 0, but 1
#markers = markers+1

## Now, mark the region of unknown with zero
#markers[unknown==255] = 0

#cv2.waitKey(0)
#cv2.destroyAllWindows()
