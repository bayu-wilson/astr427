###############
# This is a script that estimates the value of pi using MCMC 
#
# Below is an example line you could paste into the command line:
# `python Homework5_script.py 10000`
# 
# In general, run:
# `python Homework5_script.py [number of iteration for MCMC]`
###############

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda import curandom

import sys

N = int(sys.argv[1]) #int(1e3)
print("Number of iterations: " + str(N))

np.random.seed(123)
draws = np.random.uniform(-1,1,N)
index = np.linspace(-1,1,N)

a_gpu = cuda.mem_alloc(draws.nbytes)
cuda.memcpy_htod(a_gpu, draws)

gen = pycuda.curandom.XORWOWRandomNumberGenerator()
xy = gen.gen_uniform((2,N),np.float32)
xy = sum(xy**2)**0.5

M = gpuarray.sum(gpuarray.if_positive(xy-1,xy*0,xy*0+1))

pi = 4*M/N
print("The estimated value of pi is: " + str(pi))


