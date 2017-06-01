#Name: Adrian Chan 
'''
This script is a edge detection algorithm that uses convolve techniques to
detect edges
'''


import numpy as np
from bgr_to_hsv import BGRToHSV
import cv2


PREWITT_HOR = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
SOBEL_HOR = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

def MyConvolve(img, ff):
    result_V = np.zeros(img.shape)
    result_H = np.zeros(img.shape)
    result = np.zeros(img.shape)
    for i in range(1, img.shape[0] - 1):  # traverses through height of the image
        for j in range(1, img.shape[1] - 1):  # traverses through width of the image
            for kY in range(ff.shape[0]):  # traverses through width of the image
                for kX in range(ff.shape[1]):  # traverses through width of the image
                    result_V[i][j] += (img[i - 1 + kY][j - 1 + kX][0] * ff[kX][kY])
                    result_H[i][j] += (img[i - 1 + kY][j - 1 + kX][0] * ff[kY][kX])
            result[i][j] = np.hypot(result_V[i][j], result_H[i][j])

    return result

def normalise(image):
    result = np.zeros(image.shape)
    max = np.amax(image)
    min = np.amin(image)
    result = 255.0 * (image - min) / (max - min)
    return result


def EdgeThinning(img):
    result = np.zeros(img.shape)
    for i in range(img.shape[0]):  # traverses through height of the image
        for j in range(img.shape[1]):  # traverses through width of the image
            #edge case:
            if (i ==0): #top row
                if (j == 0):#top left
                    if (img[i][j][0] > img[i+1][j][0] and img[i][j][0] > img[i][j + 1][0]):
                        result[i][j] = img[i][j]
                elif (j == img.shape[1]-1):#top right
                    if (img[i][j][0] > img[i + 1][j][0] and img[i][j][0] > img[i][j - 1][0]):
                        result[i][j] = img[i][j]
                else: #top middle
                    if (img[i][j][0] > img[i + 1][j][0] and img[i][j][0] > img[i][j - 1][0] and img[i][j][0] > img[i][j + 1][0]):
                        result[i][j] = img[i][j]
            elif (i == img.shape[0]-1): #bottom row
                if (j == 0): #bottom left
                    if (img[i][j][0] > img[i-1][j][0] and img[i][j][0] > img[i][j + 1][0]):
                        result[i][j] = img[i][j]
                elif (j == img.shape[1]-1): #bottom right
                    if (img[i][j][0] > img[i - 1][j][0] and img[i][j][0] > img[i][j - 1][0]):
                        result[i][j] = img[i][j]
                else:
                    if (img[i][j][0] > img[i - 1][j][0] and img[i][j][0] > img[i][j - 1][0] and img[i][j][0] > img[i][j + 1][0]):
                        result[i][j] = img[i][j]
            #End edge case
            else: #middle cases:
                if (j == 0): #middle left
                    if (img[i][j][0] > img[i-1][j][0] and img[i][j][0] > img[i][j + 1][0] and img[i][j][0] > img[i+1][j][0]):
                        result[i][j] = img[i][j]
                elif (j == img.shape[1]-1): #middle right
                    if (img[i][j][0] > img[i - 1][j][0] and img[i][j][0] > img[i][j - 1][0] and img[i][j][0] > img[i+1][j][0]):
                        result[i][j] = img[i][j]
                else: #middle middle
                    if (img[i][j][0] > img[i + 1][j][0] and img[i][j][0] > img[i][j - 1][0] and img[i][j][0] >
                        img[i][j + 1][0] and img[i][j][0] > img[i - 1][j][0]):
                        result[i][j] = img[i][j]
    return result


#TEST 1
test1 = cv2.imread('test1.jpg')
test1_brightness = BGRToHSV(test1).value

test1_pEdge = MyConvolve(test1_brightness, PREWITT_HOR)
test1_prewitt = normalise(test1_pEdge)
cv2.imwrite("test1_prewitt.jpg", test1_prewitt)

test1_sEdge = MyConvolve(test1_brightness, SOBEL_HOR)
test1_sobel = normalise(test1_sEdge)
cv2.imwrite("test1_sobel.jpg", test1_sobel)

cv2.imwrite("test1_thinned.jpg", EdgeThinning(test1_sobel))

#TEST 2
test2 = cv2.imread('test2.jpg')
test2_brightness = BGRToHSV(test2).value

test2_pEdge = MyConvolve(test2_brightness, PREWITT_HOR)
test2_prewitt = normalise(test2_pEdge)
cv2.imwrite("test2_prewitt.jpg", test2_prewitt)

test2_sEdge = MyConvolve(test2_brightness, SOBEL_HOR)
test2_sobel = normalise(test2_sEdge)
cv2.imwrite("test2_sobel.jpg", test2_sobel)

cv2.imwrite("test2_thinned.jpg", EdgeThinning(test2_sobel))


#TEST 3
test3 = cv2.imread('test3.jpg')
test3_brightness = BGRToHSV(test3).value

test3_pEdge = MyConvolve(test3_brightness, PREWITT_HOR)
test3_prewitt = normalise(test3_pEdge)
cv2.imwrite("test3_prewitt.jpg", test3_prewitt)

test3_sEdge = MyConvolve(test3_brightness, SOBEL_HOR)
test3_sobel = normalise(test3_sEdge)
cv2.imwrite("test3_sobel.jpg", test3_sobel)

cv2.imwrite("test3_thinned.jpg", EdgeThinning(test3_sobel))


