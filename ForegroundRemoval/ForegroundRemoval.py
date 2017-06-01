#Name: Adrian Chan
'''
This Script takes a video still and removes the cars from the foreground,
leaving only the highway
'''

import cv2
import cv2.cv as cv
import numpy as np

cap = cv2.VideoCapture('traffic.mp4')

width = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CV_CAP_PROP_FPS)
frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

print width
print height
print fps
print 1.0/frameCount

_,img = cap.read()
avgImg = np.float32(img)
for fr in range(1,frameCount):
    _, img = cap.read()
    cv2.accumulateWeighted(img, avgImg, 1.0/(fr+1))
    background = cv2.convertScaleAbs(avgImg)
    cv2.imshow('img', img)
    cv2.imshow('normImg', background)  # normImg is avgImg converted into uint8
    print "fr = ", fr, " alpha = ", 1.0/fr
cv2.imwrite("traffic_background.jpg", background)
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()


