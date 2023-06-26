# -*- coding: utf-8 -*-
"""
to find the sparsity pattern of A and then the sparsity pattern of A^2
"""
import numpy as np
#%% NOTE: if A is symm A^2 is also symm



#A is symmetric, ARRAY, and diagonal has nonzero values
def find_sprs_pttn(A):
    m,n = np.shape(A)
    A_bot = np.tril(A)
    base_dict = {}
    for k in range(1,n+1):
        base_dict[k] = []
    nzro = np.nonzero(A_bot)
    
    for k in range(len(nzro[0])):
        base_dict[ nzro[0][k]+ 1 ] = base_dict[nzro[0][k]+1] + [nzro[1][k]+1]
    return(base_dict)
    



A = np.array([[True, True, False, True],
              [True, True, False, False],
              [False, False, True, True],
              [True, False, True, True]])
A_bot = np.tril(A)
test = find_sprs_pttn(A)






