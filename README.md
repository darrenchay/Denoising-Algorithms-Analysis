# Running the programs

## Pre Conditions
- Make sure that there is a Results folder in the same folder that you are running the algorithm
- Make sure that you have python and the following libraries are installed first:
    - numpy
    - scipy
    - opencv
    - matplotlib
    - tkinter

## Executing a filter
To run the program, type in 'python DenoisingAlgorithms.py'. You will be prompted to select the image you want to denoise and then input the number of the algorithm you want to choose for denoising. The result will then be stored in the Results folder after execution

Notes:
- if you want to have a comparison of the original and denoise image next to each other after running the algorithms, uncomment the last couple of lines 
- To change the values of some of the parameters of the filters, such as the radius of the gaussian filter, you will need to manually change them in the program itself (where the algorithms are being called and run)
## Executing the noise estimation algorithm
To execute the estimation algorithm, run the following command: 'python noise_variance_estimator.py' and select the processed image that you want to analyze
The value will then be output in the terminal 