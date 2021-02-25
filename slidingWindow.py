""" 
Obtained a sliding window module which can be used to create the 
kernels for mode and alpha trimmed mean filtering
from stackoverflow. Link: https://stackoverflow.com/a/11000193
"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

def cell_neighbors(arr, j, i, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()

# x = np.arange(8*8).reshape(8, 8)
# print(x)

# for i in range(8):
#     for j in range(8):
#         print("cell:(%d,%d)" % (i,j))
#         print(cell_neighbors(x, i, j, d=2))

# for d in [1, 2]:
#     for p in [(0,0), (0,1), (6,6), (8,8)]:
#         print("-- d=%d, %r" % (d, p))
#         print(cell_neighbors(x, p[0], p[1], d=d))

""" [[ 0  1  2  3  4  5  6  7] 
     [ 8  9 10 11 12 13 14 15] 
     [16 17 18 19 20 21 22 23] 
     [24 25 26 27 28 29 30 31] 
     [32 33 34 35 36 37 38 39] 
     [40 41 42 43 44 45 46 47] 
     [48 49 50 51 52 53 54 55] 
     [56 57 58 59 60 61 62 63]] """