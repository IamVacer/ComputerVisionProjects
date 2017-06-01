#Name: Adrian Chan 

'''
This code takes in the image data and converts it to HSV format and back
a histogram equaliser is also implemented to see how if image quality can be
improved
'''


import cv2
from bgr_to_hsv import BGRToHSV
from hsv_to_rgb import HSVToRGB
from histogram_equaliser import hist_equalise



concert = cv2.imread('Images/concert.jpg')
concert_hsv = BGRToHSV(concert)

sea1 = cv2.imread('Images/sea1.jpg')
sea1_hsv = BGRToHSV(concert)

sea2 = cv2.imread('Images/sea2.jpg')
sea2_hsv = BGRToHSV(concert)

concert_hsv.extract_hsv_images()
cv2.imwrite('concert_hue.jpg',concert_hsv.hue)
cv2.imwrite('concert_saturation.jpg',concert_hsv.saturation)
cv2.imwrite('concert_brightness.jpg',concert_hsv.value)

sea1_hsv.extract_hsv_images()
cv2.imwrite('sea1_hue.jpg',sea1_hsv.hue)
cv2.imwrite('sea1_saturation.jpg',sea1_hsv.saturation)
cv2.imwrite('sea1_brightness.jpg',sea1_hsv.value)

sea2_hsv.extract_hsv_images()
cv2.imwrite('sea2_hue.jpg',sea2_hsv.hue)
cv2.imwrite('sea2_saturation.jpg',sea2_hsv.saturation)
cv2.imwrite('sea2_brightness.jpg',sea2_hsv.value)


####################################### Import #######################################
concert_hue = cv2.imread('concert_hue.jpg')
concert_saturation = cv2.imread('concert_saturation.jpg')
concert_brightness = cv2.imread('concert_brightness.jpg')
concert_rgb = HSVToRGB(concert_hue, concert_saturation, concert_brightness)


sea1_hue = cv2.imread('sea1_hue.jpg')
sea1_saturation = cv2.imread('sea1_saturation.jpg')
sea1_brightness = cv2.imread('sea1_brightness.jpg')
sea1_rgb = HSVToRGB(sea1_hue, sea1_saturation, sea1_brightness)


sea2_hue = cv2.imread('sea2_hue.jpg')
sea2_saturation = cv2.imread('sea2_saturation.jpg')
sea2_brightness = cv2.imread('sea2_brightness.jpg')
sea2_rgb = HSVToRGB(sea2_hue, sea2_saturation, sea2_brightness)


cv2.imwrite('concert_hsv2rgb.jpg',concert_rgb.image_rgb)
cv2.imwrite('sea1_hsv2rgb.jpg',sea1_rgb.image_rgb)
cv2.imwrite('sea2_hsv2rgb.jpg',sea2_rgb.image_rgb)



####################################### Part 2 #######################################
####################################### Part 2 #######################################
####################################### Part 2 #######################################



concert_eq = hist_equalise(concert_brightness)
concert_rgb_eq = HSVToRGB(concert_hue, concert_saturation, concert_eq)
cv2.imwrite('concert_histeq.jpg',concert_rgb_eq.image_rgb)


sea1_eq = hist_equalise(sea1_brightness)
sea1_rgb_eq = HSVToRGB(sea1_hue, sea1_saturation, sea1_brightness)
cv2.imwrite('sea1_histeq.jpg',sea1_rgb_eq.image_rgb)


sea2_eq = hist_equalise(sea2_brightness)
sea2_rgb_eq = HSVToRGB(sea2_hue, sea2_saturation, sea2_brightness)
cv2.imwrite('sea2_histeq.jpg',sea2_rgb_eq.image_rgb)

