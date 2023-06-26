# -*- coding: utf-8 -*-
"""

this will be the MAIN FILE

"""

#%% Imports
from generate_matrices import laplacian_grid, florida_sparse
from gehat_ge_dghat_old import get_basis_vector, sparse_apprx_inv
from find_sparsity_pattern_func_old import find_sprs_pttn
import numpy as np




#%% Matrix generation and pre-stuff
A = laplacian_grid(8)
orders_of_A = 4




#%% A1 and A2 "masks"
A1 = A != 0

sprs_ptts = [A1]
for k in range(orders_of_A-1):
    sprs_ptts.append( (sprs_ptts[k])**2 )




#%% Ghat and dictionary  REPLACE SECTION AT END

pttns_nondense = []

for k in range(orders_of_A):
    pttns_nondense.append( sprs_ptts[k].todense() )

dict_pttns = []
for k in range(orders_of_A):
    dict_pttns.append( find_sprs_pttn( pttns_nondense[k] ) )










#%% connectedness and adjacency






#%%spline




#%% visualization 








