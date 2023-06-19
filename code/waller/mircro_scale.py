import numpy as np
from scipy.sparse import random, linalg
from itertools import chain, combinations
import copy
from generate_positive_definite_matrix_file import *
from dict_of_diags_file import *
from gehat_ge_dghat import *



#%%     generate positive definite matrix
A = generate_positive_definite_matrix(10)
m,n = np.shape(A)


#%%     generate the diagonals
ss = dict_of_diags(A)


#%%     calculate det(A)^(1/n)
d_A = np.linalg.det(A) ** (1 / n)


#%%     calculate the approximation from each sparsity pattern in ss
aprxs = []
for sprs_pattern in ss:
    Ghat , G, dGhat = sparse_apprx_inv(A, sprs_pattern)
    aprxs.append(dGhat)

#for large matrices ~100x100, i start getting overflow errors


#%%     count number manually
"""
entries_in_patterns = []
for dictt in ss:
    
    counting = 0
    for key in dictt:
        counting += len(dictt[key])
    entries_in_patterns.append(counting)
"""

#%%     count number specifically for diagonal dictionaries
entries_in_patterns = [m,m+m-1]
for cc in range(m-2):
    entries_in_patterns.append( entries_in_patterns[-1] + m - 2 - cc)


#%%     manually check if each set is a subset of another, and collect the index pairs of ones that are subsets



#%%     make empty adjacency matrix


#%%     with the index pairs, find difference in their sparsity as weight and store in the ajaceny matrix



#%%     calculate laplacian


#%%     visualize the graph????


#$$     minimize?????








