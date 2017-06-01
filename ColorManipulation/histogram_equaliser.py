import numpy as np

def hist_equalise(image_brightness,nbr_bins=256):
   #get image histogram
   imhist,bins = np.histogram(image_brightness.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(image_brightness.flatten(),bins[:-1],cdf)

   return im2.reshape(image_brightness.shape)
