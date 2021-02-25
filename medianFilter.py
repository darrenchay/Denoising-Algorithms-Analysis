import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# import tkinter as tkinter
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)
img = cv.imread(filename)
# blur = cv.blur(img,(5,5))
median = cv.medianBlur(img,5)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Denoised')
plt.xticks([]), plt.yticks([])
plt.show()