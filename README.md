# Content
## DenoisingAlgorithms.py
This program prompts the user to select an image that they would like to process. Then, they will be able to choose between 5 algorithms:
- Mean Filter
- Gaussian Filter
- Median Filter
- Mode Filter
- Edge-preserving filter <br />
<b>Note:</b> the mean filter did not end up being used in the experiment. <br />
After choosing the algorithm, the user is finally prompted to choose the neighbourhood size and then the image is processed and saved in the Results folder

## NoiseVarianceEstimator.py
This program prompts the user to choose an image that they want to run the fast noise variance estimator algorithm on and the variance is then output in the terminal after being processed. 
# Running the programs

## Pre Conditions
- Make sure that there is a Results folder in the same folder that you are running the algorithm
- Make sure that you have python and the following libraries are installed first (using pip install):
    - numpy
    - scipy
    - opencv
    - matplotlib
    - tkinter

## Executing a filter
To run the program, type in 'python DenoisingAlgorithms.py'. You will be prompted to select the image you want to denoise and then input the number of the algorithm you want to choose for denoising. After choosing the neighbourhood size, the result will then be stored in the Results folder after execution and the time taken to execute the algorithm will be printed out

Note: if you want to have a comparison of the original and denoise image next to each other after running the algorithms, uncomment the last couple of lines 

## Executing the noise estimation algorithm
To execute the estimation algorithm, run the following command: 'python NoiseVarianceEstimator.py' and select the processed image that you want to analyze.
The value will then be output in the terminal 

