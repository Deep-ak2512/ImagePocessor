# Considering 5 bands and they can be distinguished by width of bands

import cv2
import numpy as np
from scipy.signal import find_peaks,medfilt

def band_segment(image,max_bands=5,max_dist=50)
  if image.shape[2] == 3:
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
  else:
    gray = image

  # applying sobel kernel for finding edges along Y direction
edgeY_kernel = np.array([
    [-1,-2,-1],
    [0 , 0, 0],
    [1 ,2, 1]
].dtype=np.float32)
edgesY = cv2.filter2D(gray,ddpeth=cv2.CV_64F,kernel=edgeY_kernel)
edgesY_mag = np.abs(edges_Y)
edgesY_mag = medfilt(edgesY_mag)

# finding peaks in edge profile
fact = 0.05
peaks = find_peaks(edgesY_mag,distance = min_dist,prominance = np.max(edgesY_mag)*fact)
     
