# -*- coding: utf-8 -*-
"""
Function for approximating d(A) using the Ghat_{E^{l}} construction.

I takes in the actual matrix to be apprx. It also takes in a NOT tril 
(well it could be tril, it wouldnt error; thered be a redundant line tho)
sparsity pattern in the form of a csr array (should be boolean entries).

it returns the apprx. can return Ghat_{E^{l}} if we want it to
"""

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.sparse import linalg



def our_csr_apprx(A, pattern):
    #A is matrix to apprx d(A) and pattern is csr boolean
    m,n = A.get_shape()
    pat = sp.sparse.tril(pattern, format='csr')
    base = sp.sparse.csr_array((n,n))
    for i in range(n): #i is the row we work with in A
        nonzero_cols = pat.indices[pat.indptr[i]:pat.indptr[i+1]] #in row i, col number of nonzeros
        
        first_projection = sp.sparse.linalg.inv( (A[ nonzero_cols ,:])[:,nonzero_cols]  )  #slice according to projection then inverse then slice then slice
        
        second_projection = sp.sparse.csr_array((n,len(nonzero_cols)))
        second_projection[ nonzero_cols , 0:len(nonzero_cols) ]   =  first_projection 
        
        third_projection = base.copy()
        third_projection[ : , nonzero_cols ] = second_projection
        
        fourth_projection = sp.sparse.csr_array.transpose(third_projection[:,[i]]).tocsr()   #the e_i projection
        
        if i == 0:
            Ghat_El = (fourth_projection.copy())
        else:
            Ghat_El = sp.sparse.vstack([Ghat_El, (fourth_projection.copy())])
    Ghat_El = Ghat_El.tocsr()
    diagonals = sp.sparse.csr_matrix.diagonal(Ghat_El) **(-1/n)
    apprx = np.prod(diagonals)
        
    return(apprx, Ghat_El)
    




