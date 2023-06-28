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
from gehat_ge_dghat_old import sparse_apprx_inv



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
base = sp.sparse.csr_array((n,n))
#Ghat_El = base.copy()

#pattern= sparsity pattern    A=array to be sliced    
#for slicing, im not sure if I should switch back and forth between csr and csc formats for higher speed


for i in range(n): #i is the row we work with in A
    nonzero_cols = pattern.indices[pattern.indptr[i]:pattern.indptr[i+1]] #in row i, col number of nonzeros
    
    first_projection = sp.sparse.linalg.inv( (A[ nonzero_cols ,:])[:,nonzero_cols]  )  #slice according to projection then inverse then slice then slice
    
    second_projection = sp.sparse.csr_array((n,len(nonzero_cols)))
    second_projection[ nonzero_cols , 0:len(nonzero_cols) ]   =  first_projection 
    
    third_projection = base.copy()
    third_projection[ : , nonzero_cols ] = second_projection
    
    fourth_projection = sp.sparse.csr_array.transpose(third_projection[:,[i]]).tocsr()   #the e_i projection
    
    if i == 0:
        Ghat_El = (fourth_projection.copy())
        print(i)
    else:
        Ghat_El = sp.sparse.vstack([Ghat_El, (fourth_projection.copy())])
        print(i)
    
    
#working script for making Ghat's from the boolean A1, A2, etc.. 
#now to perform the math on the Ghat_El to find the apprx

#try the big PI method

diagonals = sp.sparse.csr_matrix.diagonal(Ghat_El) **(-1/n)
apprx = np.prod(diagonals)




#%% compare






#%% testing
test = pattern.indices[pattern.indptr[10]:pattern.indptr[10+1]]
first_projection = sp.sparse.linalg.inv( ((A[ test ,:])[:,test] ) )

second_projection = sp.sparse.csr_array((n,len(test)))
second_projection[ test , 0:len(test) ]   =  first_projection        #maybe convert to lil_matrix to change structure

third_projection = sp.sparse.csr_array((n,n))
third_projection[ : , test ] = second_projection

fourth_projection = sp.sparse.csr_array.transpose(third_projection[:,[10]]).tocsr()
Ghat_El = fourth_projection.copy()
test2222 = sp.sparse.vstack([Ghat_El,fourth_projection])

test222 = A1.getrow(1)




maybe = Ghat_El.todense()






