import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
# import tkinter as tkinter
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
from scipy import stats
import pprint

# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
# print(filename)
# img = cv.imread(filename, 0) # reading image in grayscale
img = cv.imread('Test images\groundTruths\eg3_impulseNoise.jpg',0)

# Setting up kernel
ind = 1
kernel_size = 3 + 2 * (ind % 5)
kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
pprint.pprint(kernel)
midptKernel = (int(kernel_size/2))+1

# Iterating over image
pprint.pprint(img.shape)
rows,cols = img.shape
newImg = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
        # Excluding pixels that are in the border of the kernel
        if(i >= midptKernel and j >= midptKernel and i <= (rows - midptKernel) and j <= (cols - midptKernel)):
            # print("INDEX I: " + str(i) + " INDEX J: " + str(j) + " for the pixel\n")
            # Reset kernel to 0s
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32) 
            # Populating kernel (neighbourhood) for this pixel
            for rowKern in range(kernel_size):
                for colKern in range(kernel_size):
                    neighbourIndexI = 0
                    neighbourIndexJ = 0
                    # Getting I index value of neighbourhood
                    if rowKern < midptKernel:
                        neighbourIndexI = i - (midptKernel - rowKern)
                    elif rowKern == midptKernel:
                        neighbourIndexI = i # For center row, take the i index
                    else: 
                        neighbourIndexI = i + (rowKern - midptKernel) 
                        
                    # Getting J index value of neighbourhood
                    if colKern < midptKernel:
                        neighbourIndexJ = j - (midptKernel - colKern)
                    elif colKern == midptKernel:
                        neighbourIndexJ = j # For center col, take the j index
                    else: 
                        neighbourIndexJ = j + (colKern - midptKernel) 
                        
                    # print("Neighbour I = " + str(neighbourIndexI) + " and Neighbour J = " + str(neighbourIndexJ) + ", value = " + str(img[neighbourIndexI, neighbourIndexJ]) + "\n")
                    
                    # Retrieving the value of the neighbourhood at index [rowKern, colKern] relative to pixel
                    kernel[rowKern, colKern] = img[neighbourIndexI, neighbourIndexJ]
                    
            # pprint.pprint(kernel)
            
            # Removing the 2 furthest values from the mean of the kernel
            flatKernel = kernel.flatten()
            meanKernel = int(np.mean(flatKernel))
            # print(flatKernel)
            # print(meanKernel)
            max1= flatKernel[0]
            max2 = flatKernel[0]
            indexMax1 = 0
            indexMax2 = 0
            for currIndex in range(len(flatKernel)):
                # Check if currVal > max1
                # print(abs(flatKernel[i] - meanKernel))
                if abs(flatKernel[currIndex] - meanKernel) >= abs(max1 - meanKernel):
                    # Make max2 = max1
                    max2 = max1
                    indexMax2 = indexMax1
                    # Make max1 = currVal
                    max1 = flatKernel[currIndex]
                    indexMax1 = currIndex
                # if currVal <= max1, check if currVal > max2
                else:
                    if abs(flatKernel[currIndex] - meanKernel) >= abs(max2 - meanKernel):
                        # Update max2 to new val if it is greater than max2
                        max2 = flatKernel[currIndex]
                        indexMax1 = currIndex
            # print("max1: " + str(max1) + " index1: " + str(indexMax1) + " max2: " + str(max2) + " index2: " + str(indexMax2) + "\n")
            newFlatKernel = np.delete(flatKernel, [indexMax1, indexMax2])
            # print("New Kernel: " + str(newFlatKernel))
            # Finding the mode and setting this pixel's new value to the mode
            modeKernel = stats.mode(newFlatKernel)
            # print("MODE: " + str(modeKernel[0]))
            # pprint.pprint(stats.mode(kernel.flatten()))
            newImg.itemset((i, j), modeKernel[0])
            
    if i == int(rows/2):
        print("Halfway done\n")
            


plt.subplot(121),plt.imshow(img, cmap=cm.gray),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(newImg, cmap=cm.gray),plt.title('Denoised')
plt.xticks([]), plt.yticks([])
plt.show()