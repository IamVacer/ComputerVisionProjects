import numpy as np

MAX_PIXEL_CONSTANT = 255.0
BLUE_INDEX = 0
GREEN_INDEX = 1
RED_INDEX = 2

def d_max(bgr):
    hsv = np.array([0.0, 0.0, 0.0])
    red = bgr[RED_INDEX] / MAX_PIXEL_CONSTANT
    blue = bgr[BLUE_INDEX] / MAX_PIXEL_CONSTANT
    green = bgr[GREEN_INDEX] / MAX_PIXEL_CONSTANT
    hsv[2] = max(red, blue, green)
    delta = hsv[2] - min(red, blue, green)
    if (hsv[2] == 0):
        hsv[1] = 0
    else:
        hsv[1] = delta / hsv[2]
    return hsv

def r_max(bgr):
    hsv = np.array([0.0, 0.0, 0.0])
    red = bgr[RED_INDEX] / MAX_PIXEL_CONSTANT
    blue = bgr[BLUE_INDEX] / MAX_PIXEL_CONSTANT
    green = bgr[GREEN_INDEX] / MAX_PIXEL_CONSTANT
    hsv[2] = max(red, blue, green)
    delta = hsv[2] - min(red, blue, green)
    hsv[0] = (60 * ((green - blue) / delta)) % 360 * MAX_PIXEL_CONSTANT / 360
    if (hsv[2] == 0):
        hsv[1] = 0
    else:
        hsv[1] = delta / hsv[2]
    return hsv

def g_max(bgr):
    hsv = np.array([0.0, 0.0, 0.0])
    red = bgr[RED_INDEX] / MAX_PIXEL_CONSTANT
    blue = bgr[BLUE_INDEX] / MAX_PIXEL_CONSTANT
    green = bgr[GREEN_INDEX] / MAX_PIXEL_CONSTANT
    hsv[2] = max(red, blue, green)
    delta = hsv[2] - min(red, blue, green)
    hsv[0] = (60 * ((blue - red) / delta + 2)) * MAX_PIXEL_CONSTANT / 360
    if (hsv[2] == 0):
        hsv[1] = 0
    else:
        hsv[1] = delta / hsv[2]
    return hsv


def b_max(bgr):
    hsv = np.array([0.0, 0.0, 0.0])
    red = bgr[RED_INDEX] / MAX_PIXEL_CONSTANT
    blue = bgr[BLUE_INDEX] / MAX_PIXEL_CONSTANT
    green = bgr[GREEN_INDEX] / MAX_PIXEL_CONSTANT
    hsv[2] = max(red, blue, green)
    delta = hsv[2] - min(red, blue, green)
    hsv[0] = (60 * ((red - green) / delta + 4)) * MAX_PIXEL_CONSTANT / 360
    if (hsv[2] == 0):
        hsv[1] = 0
    else:
        hsv[1] = delta / hsv[2]
    return hsv

def convert_pixel_to_hsv(bgr):
    red = bgr[RED_INDEX]
    blue = bgr[BLUE_INDEX]
    green = bgr[GREEN_INDEX]
    location = bgr.argmax(axis=0)
    if (max(red, blue, green) - min(red, blue, green) == 0):
        hsv = d_max(bgr)
    elif (location == 2):
        hsv = r_max(bgr)
    elif (location == 0):
        hsv = b_max(bgr)
    else:
        hsv = g_max(bgr)
    return hsv

class BGRToHSV:
    def __init__(self, image_array):
        self.image_array = image_array
        self.image_height = image_array.shape[0]
        self.image_width = image_array.shape[1]
        self.hue = np.zeros([image_array.shape[0], image_array.shape[1], 3])
        self.saturation = np.zeros([image_array.shape[0], image_array.shape[1],
                                    3])
        self.value = np.zeros([image_array.shape[0], image_array.shape[1], 3])

    def extract_hsv_images(self):
        for current_height in range(
                self.image_height):  # traverses through height of the image
            for current_width in range(
                    self.image_width):  # traverses through width of the image
                hsv_pixel = convert_pixel_to_hsv(self.image_array[
                                                   current_height][current_width])
                hsv_int = np.array(
                    [int(hsv_pixel[0]), int(hsv_pixel[1] * 255), int(hsv_pixel[2] * 255)])
                self.hue[current_height][current_width] = hsv_int[0]
                self.saturation[current_height][current_width] = hsv_int[1]
                self.value[current_height][current_width] = hsv_int[2]