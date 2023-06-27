# -*- coding: utf-8 -*-
"""
new file to calculate G and do the approx stuff
"""
#%%this is here just to generate the matrices to work with. will be deleted at end when made function
from generate_matrices import laplacian_grid, florida_sparse
from gehat_ge_dghat_old import sparse_apprx_inv
from find_sparsity_pattern_func_old import find_sprs_pttn
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import laplacian
import scipy as sp




dim_enter = 4
A = laplacian_grid(dim_enter)       #generate input^2 x input^2 matrix
m,n = A.get_shape()
orders_of_A = 4             #power of A to go to

A_LU = sp.sparse.linalg.splu(A)     # LU decomp
actual_det = A_LU.L.diagonal().prod() * A_LU.U.diagonal().prod()      #product of the products of the diagonal entries of the LU decomp
actual_value = actual_det**(1 / (dim_enter**2))


A1 = A != 0
sprs_ptts = [A1]
for k in range(2,orders_of_A+1):
    sprs_ptts.append( (sprs_ptts[0])**k )

viewing_mxs = []
for k in range(orders_of_A):
    viewing_mxs.append( sprs_ptts[k].todense())

#%% input are the boolean arrays of where values are


inputt = sprs_ptts[0]

pattern = sp.sparse.tril(inputt, format='csr')

A_i = 













