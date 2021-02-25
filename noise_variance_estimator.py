"""
This algorithm implementation of the noise variance estimator metric 
was found using this link: https://stackoverflow.com/a/25436112
and was adjusted accordinly to fit the needs of the current program
"""
import cv2 as cv
import numpy as np
import math
from scipy.signal import convolve2d 

from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

# Retrieving image to be measured
Tk().withdraw() 
filename = askopenfilename() 
print(filename)
img = cv.imread(filename, 0)

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma
print("Noise Variance (sigma) = ",estimate_noise(img))