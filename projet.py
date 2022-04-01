#conda install numpy=1.20
from numba import cuda
from numba.cuda.cudadrv.driver import driver
from numba.np import numpy_support as nps
import time

import numpy as np
import pandas as pd

import math, sys

import PIL
from PIL import Image


# --------------------------- DATA  -------------------------------- #

# Training data 
training = pd.read_pickle("training.pkl")
print("Training data")
print("nombres d'image: ", len(training))
print("taille de l'image",training[0][0].shape)

#exemple 
img = Image.fromarray(training[0][0])
img


# ------------------ Fonction scan exclusif  ----------------------- #

@cuda.jit
def scanGPU_MonteeDescente(argument1,m):
    
    global_id = cuda.grid(1)
    
    # Algo de montée 
    for d in range(0,m):
        step = 2**(d+1)
        step2 = 2**(d)
        
        if global_id <= N-1:
            k = global_id * step
            argument1[k + step-1] += argument1[k+step2-1]

    # Algo descente 
    argument1[N-1] = 0
    for  d in range(m-1, 0-1, -1):
        step = 2**(d+1)
        step2 = 2**(d)
        
       # global_id = cuda.grid(1)
        if  global_id <= N-1:
            k = global_id * step
            t= argument1[k+step2-1]
            argument1[k+step2-1] = argument1[k+step-1] 
            argument1[k+step-1] += t



# ------------------ fonction transpose GPU ---------------------------------#
"""
Code lachement trouvé sur internet :
https://github.com/numba/numba/blob/main/numba/cuda/kernels/transpose.py

Tentative d'implementer l'algo presenté dans l'article , mais probleme de threads non resolu voir index pour plus de detail

"""

def transpose(a, b=None):
    """Compute the transpose of 'a' and store it into 'b', if given,
    and return it. If 'b' is not given, allocate a new array
    and return that.
    This implements the algorithm documented in
    http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/
    :param a: an `np.ndarray` or a `DeviceNDArrayBase` subclass. If already on
        the device its stream will be used to perform the transpose (and to copy
        `b` to the device if necessary).
    """

    # prefer `a`'s stream if
    stream = getattr(a, 'stream', 0)

    if not b:
        cols, rows = a.shape
        strides = a.dtype.itemsize * cols, a.dtype.itemsize
        b = cuda.cudadrv.devicearray.DeviceNDArray(
            (rows, cols),
            strides,
            dtype=a.dtype,
            stream=stream)

    dt = nps.from_dtype(a.dtype)

    tpb = driver.get_device().MAX_THREADS_PER_BLOCK
    # we need to factor available threads into x and y axis
    tile_width = int(math.pow(2, math.log(tpb, 2) / 2))
    tile_height = int(tpb / tile_width)

    tile_shape = (tile_height, tile_width + 1)

    @cuda.jit
    def kernel(input, output):

        tile = cuda.shared.array(shape=tile_shape, dtype=dt)

        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bx = cuda.blockIdx.x * cuda.blockDim.x
        by = cuda.blockIdx.y * cuda.blockDim.y
        x = by + tx
        y = bx + ty

        if by + ty < input.shape[0] and bx + tx < input.shape[1]:
            tile[ty, tx] = input[by + ty, bx + tx]
        cuda.syncthreads()
        if y < output.shape[0] and x < output.shape[1]:
            output[y, x] = tile[tx, ty]

    # one block per tile, plus one for remainders
    blocks = int(b.shape[0] / tile_height + 1), int(b.shape[1] / tile_width + 1)
    # one thread per tile element
    threads = tile_height, tile_width
    kernel[blocks, threads, stream](a, b)
    
    return b



# --------------------------- Image intégrale ----------------------------- #

''' Version qui fonctionne mais mal aloué.... 
On arrive pas à la rendre plus performante, elle est trop lente par rapport à la méthode basique CPU'''

def image_integrale(image):

    padding = np.zeros((19,13))
    padding2 = np.zeros((13,32))
    image = np.concatenate([image,padding], axis=1)
    image = np.concatenate([image,padding2], axis = 0)
    image = image.astype(np.int64)
    
    N = len(image[0])
    m = int(np.log(N)/np.log(2))
    #print(image.shape)
    #print("image avant 1ere scan :", image)
    for i in range(32):
        #print(i)
        #print(image[i])
        # envoyer au device 
        scanGPU_MonteeDescente[16,16](image[i],m)
        #print(image[i])
        cuda.synchronize()
    #print(image.shape)
    #print("image apres scan", image)
    #print(image.shape)
    
    test = transpose(image)
    test = test.copy_to_host()
    #print('transpose', test)
    
    for i in range(32):
        #print(i)
        #print(image[i])
        # envoyer au device 
        scanGPU_MonteeDescente[16,16](test[i],m)
        #print(image[i])
        cuda.synchronize()
    #print('scan',test)
    scan2 = test  
    global test2
    test2 = transpose(scan2)
    test2 = test2.copy_to_host()
    test2 = test2[1:20,1:20]
    #print("image integrale ",test2 )
    return test2



start = time.time()
image_integrale(img)
end = time.time()
elapsed = end - start
print(f'Temps d\'execution GPU : {elapsed:.2}s')


# fonction utilisé dans le code github , le but etant que notre fonction soit plus rapide ce que je doute 
def integral_imageCPU(image):
    """
    Computes the integral image representation of a picture. The integral image is defined as following:
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Where s(x, y) is a cumulative row-sum, ii(x, y) is the integral image, and i(x, y) is the original image.
    The integral image is the sum of all pixels above and left of the current pixel
      Args:
        image : an numpy array with shape (m, n)
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii


start = time.time()
integral_imageCPU(img)
end = time.time()
elapsed = end - start
print(f'Temps d\'execution CPU : {elapsed:.2}s')

# La fonction scan integrale est 50 fois plus longue que la fonction cpu du code,
# la mission est un echeque ... 