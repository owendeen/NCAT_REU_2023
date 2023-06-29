# -*- coding: utf-8 -*-
"""
to find the sparsity pattern of A and then the sparsity pattern of A^2
"""
from generate_positive_definite_matrix_file import *
import numpy as np
from generate_positive_definite_matrix_file import *
#%% NOTE: if A is symm A^2 is also symm

#A is symmetric and diagonal has nonzero values
A = np.array([[1, 4, 0, 5],
              [4, 1, 0, 0],
              [0, 0, 2, 2],
              [5, 0, 2, 5]])
m,n = np.shape(A)


#%%since A is symmetric, take just the bottom half

A_bot = A.copy()
for k in range(m):
    for kk in range(n):
        if k < kk:
            A_bot[k,kk] = 0

#%% setup dictionary by making the diagonal
base_dict = {}
for k in range(n):
    base_dict[k] = []


#%%take indices of nonzeros. if you want the pattern of the full thing, and not the lower, replace A_bot with A

nzro = np.nonzero(A_bot)

sparsity_percent = ((n**2 - len(nzro[0])) / n**2)*100

#%% this is the sparsity pattern of the lower half

for k in range(len(nzro[0])):
    base_dict[ nzro[0][k] ] = base_dict[nzro[0][k]] + [nzro[1][k]]


#now to make this whole thing into a function to be called for finding the sparsity pattern of a given A











