#Name: Adrian Chan
#Matric Number: A0122061

import numpy as np
import numpy.linalg as la
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches



sobel_Hor = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
step_small = 1
step_norm = 10

def MyConvolve(img, ff):
    result_V = np.zeros(img.shape)
    result_H = np.zeros(img.shape)
    result = np.zeros(img.shape)
    for i in range(1, img.shape[0] - 1):  # traverses through height of the image
        for j in range(1, img.shape[1] - 1):  # traverses through width of the image
            for kY in range(ff.shape[0]):  # traverses through width of the image
                for kX in range(ff.shape[1]):  # traverses through width of the image
                    result_V[i][j] += (img[i - 1 + kY][j - 1 + kX] * ff[kX][kY])
                    result_H[i][j] += (img[i - 1 + kY][j - 1 + kX] * ff[kY][kX])
            result[i][j] = np.hypot(result_V[i][j], result_H[i][j])

    return result_V, result_H, result

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

def gauss_kernels(size,sigma=1.0):
    ## returns a 2d gaussian kernel
    if size<3:
        size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    kernel_sum = kernel.sum()
    if not sum==0:
        kernel = kernel/kernel_sum
    return kernel
def harriscalculator(w_xx, w_xy, w_yy, step):
    k = 0.06
    height = len(w_xx)/step
    width = len(w_xx[0]) / step
    response_matrix = np.zeros([height, width])
    max_response = 0
    for i in range (height):
        for j in range(width):
            w = np.matrix([[w_xx[i*step][j*step], w_xy[i*step][j*step]], [w_xy[i*step][j*step], w_yy[i*step][j*step]]])
            det = np.linalg.det(w)
            trace = np.trace(w)
            response = det - (k * trace * trace)
            response_matrix[i][j] = response
            max_response = max(max_response, response)
    return max_response, response_matrix

def drawBoxes(img, response, max, name, step):
    plotImg = plt.figure()
    axImg = plotImg.add_subplot(111, aspect = 'equal')
    axImg.imshow(img)
    #plt.hold(True)
    rows = []
    cols = []
    for i in range(len(response)):
        for j in range(len(response[0])):
            if (response[i][j]>= 0.1*max ):
                '''
                rows.append(i*step)
                cols.append(j*step)
                '''
                p = patches.Rectangle(
                    (j*step-9/2, i*step-9/2), 9, 9, fill=False,
                    edgecolor="#0000ff"
                )
                axImg.add_patch(p)
    plotImg.savefig(name, dpi=90, bbox_inches='tight')
    #plt.scatter(cols, rows, color='blue', marker='s', facecolors='none', s= 89)
    #plt.show()

checker = cv2.imread('checker.jpg',0)
checker_C = cv2.imread('checker.jpg')
b,g,r = cv2.split(checker_C)       # get b,g,r
checker_C = cv2.merge([r,g,b])     # switch it to rgb

flower = cv2.imread('flower.jpg',0)
flower_C = cv2.imread('flower.jpg')
b,g,r = cv2.split(flower_C)       # get b,g,r
flower_C = cv2.merge([r,g,b])     # switch it to rgb

test1 = cv2.imread('test1.jpg',0)
test1_C = cv2.imread('test1.jpg')
b,g,r = cv2.split(test1_C)       # get b,g,r
test1_C = cv2.merge([r,g,b])     # switch it to rgb

test2 = cv2.imread('test2.jpg',0)
test2_C = cv2.imread('test2.jpg')
b,g,r = cv2.split(test2_C)       # get b,g,r
test2_C = cv2.merge([r,g,b])     # switch it to rgb

test3 = cv2.imread('test3.jpg',0)
test3_C = cv2.imread('test3.jpg')
b,g,r = cv2.split(test3_C)       # get b,g,r
test3_C = cv2.merge([r,g,b])     # switch it to rgb

gauss_kernel = gauss_kernels(3,1)

checker_gy, checker_gx, checker_gxy = MyConvolve(checker, sobel_Hor)
flower_gx, flower_gy, flower_gxy = MyConvolve(flower, sobel_Hor)
test1_gx, test1_gy, test1_gxy = MyConvolve(test1, sobel_Hor)
test2_gx, test2_gy, test2_gxy = MyConvolve(test2, sobel_Hor)
test3_gx, test3_gy, test3_gxy = MyConvolve(test3, sobel_Hor)
print "Sobel convolved"
print checker_gy
cv2.imwrite("checker_gy.jpg", checker_gy)

checker_ixx = checker_gx * checker_gx
checker_ixy = checker_gx * checker_gy
checker_iyy = checker_gy * checker_gy

checker_wx, checker_wy, checker_wxx = MyConvolve(checker_ixx, gauss_kernel)
checker_wx, checker_wy, checker_wxy = MyConvolve(checker_ixy, gauss_kernel)
checker_wx, checker_wy, checker_wyy = MyConvolve(checker_iyy, gauss_kernel)
print "Checker Gauss Convolved"


flower_ixx = flower_gx * flower_gx
flower_ixy = flower_gx * flower_gy
flower_iyy = flower_gy * flower_gy

flower_wx, flower_wy, flower_wxx = MyConvolve(flower_ixx, gauss_kernel)
flower_wx, flower_wy, flower_wxy = MyConvolve(flower_ixy, gauss_kernel)
flower_wx, flower_wy, flower_wyy = MyConvolve(flower_iyy, gauss_kernel)
print "Flower Gauss Convolved"


test1_ixx = test1_gx * test1_gx
test1_ixy = test1_gx * test1_gy
test1_iyy = test1_gy * test1_gy

test1_wx, test1_wy, test1_wxx = MyConvolve(test1_ixx, gauss_kernel)
test1_wx, test1_wy, test1_wxy = MyConvolve(test1_ixy, gauss_kernel)
test1_wx, test1_wy, test1_wyy = MyConvolve(test1_iyy, gauss_kernel)
print "Test1 Gauss Convolved"


test2_ixx = test2_gx * test2_gx
test2_ixy = test2_gx * test2_gy
test2_iyy = test2_gy * test2_gy

test2_wx, test2_wy, test2_wxx = MyConvolve(test2_ixx, gauss_kernel)
test2_wx, test2_wy, test2_wxy = MyConvolve(test2_ixy, gauss_kernel)
test2_wx, test2_wy, test2_wyy = MyConvolve(test2_iyy, gauss_kernel)
print "Test2 Gauss Convolved"

test3_ixx = test3_gx * test3_gx
test3_ixy = test3_gx * test3_gy
test3_iyy = test3_gy * test3_gy

test3_wx, test3_wy, test3_wxx = MyConvolve(test3_ixx, gauss_kernel)
test3_wx, test3_wy, test3_wxy = MyConvolve(test3_ixy, gauss_kernel)
test3_wx, test3_wy, test3_wyy = MyConvolve(test3_iyy, gauss_kernel)
print "Test3 Gauss Convolved"

checker_small_max_response, checker_small_response = harriscalculator(checker_wxx, checker_wxy, checker_wyy, step_small)
drawBoxes(checker_C, checker_small_response, checker_small_max_response, 'checker_corners_1.jpg', step_small)
checker_norm_max_response, checker_norm_response = harriscalculator(checker_wxx, checker_wxy, checker_wyy, step_norm)
drawBoxes(checker_C, checker_norm_response, checker_norm_max_response, 'checker_corners_10.jpg', step_norm)

print "Checkers Completed"


flower_small_max_response, flower_small_response = harriscalculator(flower_wxx, flower_wxy, flower_wyy, step_small)
drawBoxes(flower_C, flower_small_response, flower_small_max_response, 'flower_corners_1.jpg', step_small)
flower_norm_max_response, flower_norm_response = harriscalculator(flower_wxx, flower_wxy, flower_wyy, step_norm)
drawBoxes(flower_C, flower_norm_response, flower_norm_max_response, 'flower_corners_10.jpg', step_norm)

print "Flowers Completed"


test1_small_max_response, test1_small_response = harriscalculator(test1_wxx, test1_wxy, test1_wyy, step_small)
drawBoxes(test1_C, test1_small_response, test1_small_max_response, 'test1_corners_1.jpg', step_small)
test1_norm_max_response, test1_norm_response = harriscalculator(test1_wxx, test1_wxy, test1_wyy, step_norm)
drawBoxes(test1_C, test1_norm_response, test1_norm_max_response, 'test1_corners_10.jpg', step_norm)

print "Test1 Completed"

test2_small_max_response, test2_small_response = harriscalculator(test2_wxx, test2_wxy, test2_wyy, step_small)
drawBoxes(test2_C, test2_small_response, test2_small_max_response, 'test2_corners_1.jpg', step_small)
test2_norm_max_response, test2_norm_response = harriscalculator(test2_wxx, test2_wxy, test2_wyy, step_norm)
drawBoxes(test2_C, test2_norm_response, test2_norm_max_response, 'test2_corners_10.jpg', step_norm)

print "Test2 Completed"


test3_small_max_response, test3_small_response = harriscalculator(test3_wxx, test3_wxy, test3_wyy, step_small)
drawBoxes(test3_C, test3_small_response, test3_small_max_response, 'test3_corners_1.jpg', step_small)
test3_norm_max_response, test3_norm_response = harriscalculator(test3_wxx, test3_wxy, test3_wyy, step_norm)
drawBoxes(test3_C, test3_norm_response, test3_norm_max_response, 'test3_corners_10.jpg', step_norm)

print "Test3 Completed"