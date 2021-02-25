""" 
Name: Darren Chay Loong
ID: 1049254
Course: CIS*4720
Assignment 2

Description: This is a denoising algorithm runner which prompts the user to choose an image file to be denoised and
             choose the algorithm to denoise the image. Then the appropriate algorithm is called and executed and the
             image is then stored in the Results folder
"""
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
import time

# ALGORITHMS


def meanFilter(img):
    mean = cv.blur(img, (5, 5))
    return mean


def medianFilter(img):
    median = cv.medianBlur(img, 5)
    return median


def gaussianFilter(img):
    gaussianImg = cv.GaussianBlur(img, (5, 5), 5)
    return gaussianImg


def modeFilter(img):
    # Setting up kernel
    ind = 1
    kernel_size = 3 + 2 * (ind % 5)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    midptKernel = (int(kernel_size/2))+1

    # Iterating over image
    pprint.pprint(img.shape)
    rows, cols = img.shape
    newImg = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            # Excluding pixels that are in the border of the kernel
            if(i >= midptKernel and j >= midptKernel and i <= (rows - midptKernel) and j <= (cols - midptKernel)):

                # Populating kernel (neighbourhood) for this pixel
                for rowKern in range(kernel_size):
                    for colKern in range(kernel_size):
                        neighbourIndexI = 0
                        neighbourIndexJ = 0
                        # Getting I index value of neighbourhood
                        if rowKern < midptKernel:
                            neighbourIndexI = i - (midptKernel - rowKern)
                        elif rowKern == midptKernel:
                            neighbourIndexI = i  # For center row, take the i index
                        else:
                            neighbourIndexI = i + (rowKern - midptKernel)

                        # Getting J index value of neighbourhood
                        if colKern < midptKernel:
                            neighbourIndexJ = j - (midptKernel - colKern)
                        elif colKern == midptKernel:
                            neighbourIndexJ = j  # For center col, take the j index
                        else:
                            neighbourIndexJ = j + (colKern - midptKernel)

                        # Retrieving the value of the neighbourhood at index [rowKern, colKern] relative to pixel
                        kernel[rowKern, colKern] = img[neighbourIndexI,
                                                       neighbourIndexJ]

                # pprint.pprint(kernel)

                # Removing the 2 furthest values from the mean of the kernel
                flatKernel = kernel.flatten()
                meanKernel = int(np.mean(flatKernel))
                # print(flatKernel)
                # print(meanKernel)
                max1 = flatKernel[0]
                max2 = flatKernel[0]
                indexMax1 = 0
                indexMax2 = 0
                for currIndex in range(len(flatKernel)):
                    # Check if currVal > max1
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

                # Delete the 2 values
                newFlatKernel = np.delete(flatKernel, [indexMax1, indexMax2])
                # Finding the median and setting this pixel's new value to the median
                medianVal = int(np.median(newFlatKernel))
                # print("Median: " + str(medianVal))
                # pprint.pprint(stats.mode(kernel.flatten()))
                newImg.itemset((i, j), medianVal)
            # For values that are at the border, no filtering is applied
            else:
                newImg.itemset((i, j), img[i, j])
        if i % 10 == 0:
            print("Processing row: ", i)
    return newImg


def nagaoMatsuyamaFilter(img):
    # Setting up a 5x5 kernel
    kernel_size = 5
    kernel = np.zeros((kernel_size, kernel_size),
                      dtype=np.float32)  # Create kernel
    midptKernel = (int(kernel_size/2))+1

    # Iterating over image
    pprint.pprint(img.shape)
    rows, cols = img.shape
    # Creating new image template
    newImg = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            # Excluding pixels that are in the border of the kernel
            if(i > midptKernel and j > midptKernel and i < (rows - midptKernel) and j < (cols - midptKernel)):

                # Populating kernel (neighbourhood) for this pixel
                for rowKern in range(kernel_size):
                    for colKern in range(kernel_size):
                        neighbourIndexI = 0
                        neighbourIndexJ = 0
                        # Getting I index value of current neighbourhood pixel
                        if rowKern < midptKernel:
                            neighbourIndexI = i - (midptKernel - rowKern)
                        elif rowKern == midptKernel:
                            neighbourIndexI = i  # For center row, take the i index
                        else:
                            neighbourIndexI = i + (rowKern - midptKernel)

                        # Getting J index value of current neighbourhood pixel
                        if colKern < midptKernel:
                            neighbourIndexJ = j - (midptKernel - colKern)
                        elif colKern == midptKernel:
                            neighbourIndexJ = j  # For center col, take the j index
                        else:
                            neighbourIndexJ = j + (colKern - midptKernel)

                        # Retrieving the value of the neighbourhood pixel at index [rowKern, colKern] relative to current pixel
                        kernel[rowKern, colKern] = img[neighbourIndexI,
                                                       neighbourIndexJ]
                # pprint.pprint(kernel)

                # Creating the sections
                centerSection = np.array([kernel[1, 1], kernel[1, 2], kernel[1, 3], kernel[2, 1],
                                          kernel[2, 2], kernel[2, 3], kernel[3, 1], kernel[3, 2], kernel[3, 3]], dtype=np.float32)
                topLeftSection = np.array([kernel[0, 0], kernel[0, 1], kernel[1, 0],
                                           kernel[1, 1], kernel[1, 2], kernel[2, 1], kernel[2, 2]], dtype=np.float32)
                topSection = np.array([kernel[0, 1], kernel[0, 2], kernel[0, 3], kernel[1, 1],
                                       kernel[1, 2], kernel[1, 3], kernel[2, 2]], dtype=np.float32)
                topRightSection = np.array([kernel[0, 3], kernel[0, 4], kernel[1, 2],
                                            kernel[1, 3], kernel[1, 4], kernel[2, 2], kernel[2, 3]], dtype=np.float32)
                rightSection = np.array([kernel[1, 3], kernel[1, 4], kernel[2, 2], kernel[2, 3],
                                         kernel[2, 4], kernel[3, 3], kernel[3, 4]], dtype=np.float32)
                bottomRightSection = np.array([kernel[2, 2], kernel[2, 3], kernel[3, 2],
                                               kernel[3, 3], kernel[3, 4], kernel[4, 3], kernel[4, 4]], dtype=np.float32)
                bottomSection = np.array([kernel[2, 2], kernel[3, 1], kernel[3, 2],
                                          kernel[3, 3], kernel[4, 1], kernel[4, 2], kernel[4, 3]], dtype=np.float32)
                bottomLeftSection = np.array([kernel[2, 1], kernel[2, 2], kernel[3, 0],
                                              kernel[3, 1], kernel[3, 2], kernel[4, 0], kernel[4, 1]], dtype=np.float32)
                leftSection = np.array([kernel[1, 0], kernel[1, 1], kernel[2, 0], kernel[2, 1],
                                        kernel[2, 2], kernel[3, 0], kernel[3, 1]], dtype=np.float32)

                sections = [centerSection, topLeftSection, topSection, topRightSection,
                            rightSection, bottomRightSection, bottomSection, bottomLeftSection, leftSection]

                # pprint.pprint(sections)
                # Finding the section with the least variance and saving the mean for that section
                currVar = 1000000
                minVar = currVar
                minSectionMean = np.mean(sections[0])
                for section in sections:
                    currVar = np.var(section)
                    if currVar < minVar:
                        minVar = currVar
                        minSectionMean = np.mean(section)
                    # print("Curr Var = %d, CurrMean = %d, minVar = %d" % (currVar, minSectionMean, minVar))

                # Setting processed pixel in new image
                newImg.itemset((i, j), minSectionMean)
            # For values that are at the border, no filtering is applied
            else:
                # print("x: %d, y=%d, val=%d"%(i,j, img[i,j]))
                newImg.itemset((i, j), img[i, j])

        if i % 10 == 0:
            print("Processing row: ", i)
    return newImg


# Reading the file
Tk().withdraw()
filename = askopenfilename()
print("filename", filename)
img = cv.imread(filename, 0)  # reading image in grayscale

# Check if file was selected
if not filename:
    print("error: no file selected")
    exit()

# Choosing which algorithm to run
algorithmNo = 0
while int(algorithmNo) not in [1, 2, 3, 4, 5]:
    print("***************************")
    print("\tAlgorithms:")
    print("***************************")
    print("Mean Filter:\t\t", 1)
    print("Gaussian Filter:\t", 2)
    print("Median Filter:\t\t", 3)
    print("Mode Filter:\t\t", 4)
    print("Nagao-Matsuyama Filter:\t", 5)
    print("***************************")
    algorithmNo = input("Enter the number of the algorithm you want to run:")
    if not algorithmNo.isnumeric():
        algorithmNo = 0

algorithmNo = int(algorithmNo)

# Saving the name of the output image file
filenameSplit = filename.split("/")

# Starting timer
startTime = time.time()

# Running the appropriate algorithm
if algorithmNo is 1:
    newFileName = 'Results/mean_output_' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Mean Filter\n====================")
    processedImage = meanFilter(img)
elif algorithmNo is 2:

    newFileName = 'Results/gaussian_output_' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Gaussian Filter\n====================")
    processedImage = gaussianFilter(img)
elif algorithmNo is 3:

    newFileName = 'Results/median_output_' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Median Filter\n====================")
    processedImage = medianFilter(img)
elif algorithmNo is 4:

    newFileName = 'Results/mode_output_' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Mode Filter\n====================")
    processedImage = modeFilter(img)
else:
    newFileName = 'Results/nagao-matsuyama_output_' + \
        filenameSplit[len(filenameSplit)-1]
    print("Output File: ", newFileName)

    print("====================\nRunning Nagao-Matsuyama Filter\n====================")
    processedImage = nagaoMatsuyamaFilter(img)

# Ending timer
endTime = time.time()
timeTaken = round(endTime - startTime, 3)
print("Time taken: ", timeTaken, " seconds")

# Saving processed image
cv.imwrite(newFileName, processedImage)

# UNCOMMENT HERE FOR IMAGE COMPARISONS
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(processedImage),plt.title('Denoised')
# plt.xticks([]), plt.yticks([])
# plt.show()
