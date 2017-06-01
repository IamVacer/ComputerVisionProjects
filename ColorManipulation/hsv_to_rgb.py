import numpy as np

def convert_pixel_to_rgb(hsv):
    rgb = np.array([0.0, 0.0, 0.0])
    h = hsv[0]*360.0/255.0
    s = hsv[1]/255.0
    v = hsv[2]/255.0
    c = v * s
    x = c * (1 - abs((h/60)%2 - 1))
    m = v - c
    if (0 <= h < 60):
        rgb[2] = c
        rgb[1] = x
    elif (60 <= h < 120):
        rgb[2] = x
        rgb[1] = c
    elif (120 <= h < 180):
        rgb[0] = x
        rgb[1] = c
    elif (180 <= h < 240):
        rgb[0] = c
        rgb[1] = x
    elif (240 <= h < 300):
        rgb[2] = x
        rgb[0] = c
    else:
        rgb[0] = x
        rgb[2] = c
    for i in range(len(rgb)):
        rgb[i] = (rgb[i] + m)*255
    return rgb



class HSVToRGB:
    def __init__(self, image_hue, image_saturation, image_brightness):
        self.image_hue = image_hue
        self.image_saturation = image_saturation
        self.image_brightness = image_brightness
        self.image_height = image_hue.shape[0]
        self.image_width = image_hue.shape[1]
        self.image_rgb = np.zeros([image_hue.shape[0], image_hue.shape[1], 3])
        self.develop_rgb_mages()

    def develop_rgb_mages(self):
        hsv_pixel = np.array([0.0, 0.0, 0.0])
        for current_height in range(
                self.image_height):  # traverses through height of the image
            for current_width in range(
                    self.image_width):  # traverses through width of the image
                hsv_pixel[0] = self.image_hue[current_height][current_width][0]
                hsv_pixel[1] = self.image_saturation[current_height][current_width][0]
                hsv_pixel[2] = self.image_brightness[current_height][current_width][0]
                self.image_rgb[current_height][current_width] = convert_pixel_to_rgb(hsv_pixel)