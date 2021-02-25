# Image Processing Libraries
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy import stats

# Utility Libraries
from tkinter import Tk     
from tkinter.filedialog import askopenfilename
import pprint

# Reading the file
Tk().withdraw() 
filename = askopenfilename() 
# print(filename)
img = cv.imread(filename, 0) # reading image in grayscale

# Saving the name of the output image file
filenameSplit = filename.split("/")
newFileName = 'Results/nagao-matsuyama_output_algo_' + filenameSplit[len(filenameSplit)-1]
# img = cv.imread('Test images\groundTruths\eg3_impulseNoise.jpg',0)

# Setting up kernel
kernel_size = 5
kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32) # Create kernel
midptKernel = (int(kernel_size/2))+1

# Iterating over image
pprint.pprint(img.shape)
rows,cols = img.shape
newImg = np.zeros((rows, cols), dtype=np.float32) # Creating new image template
for i in range(rows):
    for j in range(cols):
        # Excluding pixels that are in the border of the kernel
        if(i > midptKernel and j > midptKernel and i < (rows - midptKernel) and j < (cols - midptKernel)):
            # print("INDEX I: " + str(i) + " INDEX J: " + str(j) + " for the pixel\n")
            # Reset kernel to 0s
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32) 
            
            # Populating kernel (neighbourhood) for this pixel
            for rowKern in range(kernel_size):
                for colKern in range(kernel_size):
                    neighbourIndexI = 0
                    neighbourIndexJ = 0
                    # Getting I index value of current neighbourhood pixel
                    if rowKern < midptKernel:
                        neighbourIndexI = i - (midptKernel - rowKern)
                    elif rowKern == midptKernel:
                        neighbourIndexI = i # For center row, take the i index
                    else: 
                        neighbourIndexI = i + (rowKern - midptKernel) 
                        
                    # Getting J index value of current neighbourhood pixel
                    if colKern < midptKernel:
                        neighbourIndexJ = j - (midptKernel - colKern)
                    elif colKern == midptKernel:
                        neighbourIndexJ = j # For center col, take the j index
                    else: 
                        neighbourIndexJ = j + (colKern - midptKernel) 
                        
                    # print("Neighbour I = " + str(neighbourIndexI) + " and Neighbour J = " + str(neighbourIndexJ) + ", value = " + str(img[neighbourIndexI, neighbourIndexJ]) + "\n")
                    
                    # Retrieving the value of the neighbourhood pixel at index [rowKern, colKern] relative to current pixel
                    kernel[rowKern, colKern] = img[neighbourIndexI, neighbourIndexJ]
            # pprint.pprint(kernel)
            # Creating the sections
            centerSection = np.array([kernel[1,1], kernel[1,2], kernel[1,3], kernel[2,1], kernel[2,2], kernel[2,3], kernel[3,1], kernel[3,2], kernel[3,3]], dtype=np.float32)
            topLeftSection = np.array([kernel[0,0], kernel[0,1], kernel[1,0], kernel[1,1], kernel[1,2], kernel[2,1], kernel[2,2]], dtype=np.float32)
            topSection = np.array([kernel[0,1], kernel[0,2], kernel[0,3], kernel[1,1], kernel[1,2], kernel[1,3], kernel[2,2]], dtype=np.float32)
            topRightSection = np.array([kernel[0,3], kernel[0,4], kernel[1,2], kernel[1,3], kernel[1,4], kernel[2,2], kernel[2,3]], dtype=np.float32)
            rightSection = np.array([kernel[1,3], kernel[1,4], kernel[2,2], kernel[2,3], kernel[2,4], kernel[3,3], kernel[3,4]], dtype=np.float32)
            bottomRightSection = np.array([kernel[2,2], kernel[2,3], kernel[3,2], kernel[3,3], kernel[3,4], kernel[4,3], kernel[4,4]], dtype=np.float32)
            bottomSection = np.array([kernel[2,2], kernel[3,1], kernel[3,2], kernel[3,3], kernel[4,1], kernel[4,2], kernel[4,3]], dtype=np.float32)
            bottomLeftSection = np.array([kernel[2,1], kernel[2,2], kernel[3,0], kernel[3,1], kernel[3,2], kernel[4,0], kernel[4,1]], dtype=np.float32)
            leftSection = np.array([kernel[1,0], kernel[1,1], kernel[2,0], kernel[2,1], kernel[2,2], kernel[3,0], kernel[3,1]], dtype=np.float32)
            
            sections = [centerSection, topLeftSection, topSection, topRightSection, rightSection, bottomRightSection, bottomSection, bottomLeftSection, leftSection]
            
            # pprint.pprint(sections)
            
            currVar = 1000000
            minVar = currVar
            minSectionMean = np.mean(sections[0])
            for section in sections:
                currVar = np.var(section)
                if currVar < minVar:
                    minVar = currVar
                    minSectionMean = np.mean(section)
                # print("Curr Var = %d, CurrMean = %d, minVar = %d" % (currVar, minSectionMean, minVar))
            
            newImg.itemset((i, j), minSectionMean)
        # For values that are at the border, no filtering is applied
        else:
            # print("x: %d, y=%d, val=%d"%(i,j, img[i,j]))
            newImg.itemset((i, j), img[i,j])
            
    if i%10 == 0:
        print("Row: %d" % i)
        
            
cv.imwrite(newFileName,newImg)

plt.subplot(121),plt.imshow(img, cmap=cm.gray),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(newImg, cmap=cm.gray),plt.title('Denoised')
plt.xticks([]), plt.yticks([])
plt.show()