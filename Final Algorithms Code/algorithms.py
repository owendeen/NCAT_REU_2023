'''

alg1 : compute powers until convergence
alg2 : compute spline approx until convergence
alg3 : compute all powers up to a fixed degree simultaneously
    
'''

#imports
import numpy as np
import scipy as sp
from dotdict import dotdict
from copy import copy


def splineApprox(x,y,jump=1.5):
    
    # add approx point
    #x += [x[-1]*jump] # fixed percentage jump
    x += [ x[-1] + jump*(x[-1]-x[-2]) ]
    
    
    # define adjacency,laplacian matrix
    Adjacency = np.diag(x[1:],1)-np.diag(x[:-1],1)
    Adjacency[Adjacency!=0] = 1/Adjacency[Adjacency!=0]
    Adjacency += Adjacency.T
    Laplacian = np.diag(np.sum(Adjacency,axis=0))-Adjacency

    # compute approximation
    denom = Laplacian[:,-1]@Laplacian[:,-1]
    val = -Laplacian[:,-1]@(Laplacian[:,:-1]@y)/denom
    return(val)



def logdet1(A,tol=1.05,max_power=4):
    
    # initial values
    n = A.shape[0]
    ratio = np.inf
    E = [A!=0]  # sparsity pattern full
    power = 1
    
    # initialize output
    res = {}
    approx = []
    sparsity = []
    
    while ratio > tol and power<=max_power:
        #print(power)
        if power == 2:
            E += [E[0]@E[0]]
        elif power > 2:
            E[1] = E[0]@E[-1]
        
        pattern = sp.sparse.tril(E[-1], format='csr')
        sparsity += [np.sum(pattern)/(n*(n+1)/2)]
        
        det = 0
        for i in range(n):
            nonzero_cols = pattern.indices[pattern.indptr[i]:pattern.indptr[i+1]] 
            lennz = len(nonzero_cols)
            if lennz == 1:
                val = A[i,i]**0.5
            else:
                Ai = (A[ nonzero_cols ,:])[:,nonzero_cols]         
                Li = sp.linalg.cholesky(Ai.todense())
                val = Li[-1,-1]
            det += 2*np.log(val)
        approx += [det]    
        
        if power >= 2:
            ratio = approx[-2]/approx[-1]
        power += 1
    
    # returned values
    res['approx'] = approx
    res['sparse'] = sparsity
    res = dotdict(res)
    return(res)


def logdet2(A,tol=1.1,max_power=5):
    
    # initial values
    n = A.shape[0]
    ratio = np.inf
    E = [A!=0]  # sparsity pattern full
    power = 1
    
    # initialize output
    res = {}
    approx = []
    spline = []
    sparsity = []
    
    while ratio > tol and power<=max_power:
        #print(power)
        if power == 2:
            E += [E[0]@E[0]]
        elif power > 2:
            E[1] = E[0]@E[-1]
        
        pattern = sp.sparse.tril(E[-1], format='csr')
        sparsity += [sp.sum(pattern)/(n*(n+1)/2)]
        
        det = 0
        for i in range(n):
            nonzero_cols = pattern.indices[pattern.indptr[i]:pattern.indptr[i+1]] 
            lennz = len(nonzero_cols)
            if lennz == 1:
                val = A[i,i]**0.5
            else:
                Ai = (A[ nonzero_cols ,:])[:,nonzero_cols]         
                Li = sp.linalg.cholesky(Ai.todense())
                val = Li[-1,-1]
            det += 2*np.log(val)
        approx += [det]    
        
        if power >= 2:
            spline += [splineApprox(copy(sparsity),approx)]
        if len(spline)>=2: 
            ratio = max(spline[-2],spline[-1])/min(spline[-2],spline[-1])
        power += 1
    # returned items
    res['approx'] = approx
    res['sparse'] = sparsity
    res['spline'] = spline
    res = dotdict(res)
    return(res)


def logdet3(A,max_power=5):
    
    # initial values
    n = A.shape[0]
    E = A!=0  # sparsity pattern full
    
    # initialize output
    res = {}
    approx = [0 for i in range(max_power)]
    sparsity = [0 for i in range(max_power)]
    for i in range(n):
        row = E[i,:]
        #print(row)
        
        # compute row i for each power j
        for j in range(max_power):
            
            #pattern = copy(row)
            #nonzero_cols_full =pattern.indices[pattern.indptr[0]:pattern.indptr[0+1]]
            #nonzero_cols = [kk for kk in nonzero_cols_full if kk <= i]
            #nonzero_cols = sorted(nonzero_cols)
            
            
            pattern = row[0,:i+1].todense()
            nonzero_cols = np.flatnonzero(pattern)
            
            
            #pattern = copy(row[0,:i+1])
            #nonzero_cols = pattern.indices[pattern.indptr[0]:pattern.indptr[0+1]] 
            #nonzero_cols = sorted(nonzero_cols)
            lennz = len(nonzero_cols)
            if lennz == 1:
                val = A[i,i]**0.5
            else:
                Ai = (A[ nonzero_cols ,:])[:,nonzero_cols]         
                Li = sp.linalg.cholesky(Ai.todense())
                val = Li[-1,-1]
            approx[j] += 2*np.log(val)
            sparsity[j] += lennz
            row = row@E
            
    # compute sparsity as percentage
    sparsity = [s/(n*(n+1)/2) for s in sparsity]
    
    # returned values
    res['approx'] = approx
    res['sparse'] = sparsity
    res = dotdict(res)
    return(res)

def sparse_inv(A,max_power=4):
    
    # initial values
    n = A.shape[0]
    E = [A!=0]  # sparsity pattern full
    
    power = 1
    while power<max_power:
        if power == 1:
            E += [E[0]@E[0]]
        elif power >= 2:
            E[1] = E[0]@E[-1]
        power +=1
    
    # initialize output
    res = {}
    sparsity = 0
    
    # compute pattern
    pattern = sp.sparse.tril(E[-1], format='csr')
    sparsity = np.sum(pattern)/(n*(n+1)/2)
    
    det = 0
    for i in range(n):
        nonzero_cols = pattern.indices[pattern.indptr[i]:pattern.indptr[i+1]] 
        lennz = len(nonzero_cols)
        if lennz == 1:
            val = A[i,i]**0.5
        else:
            Ai = (A[ nonzero_cols ,:])[:,nonzero_cols]         
            Li = sp.linalg.cholesky(Ai.todense())
            val = Li[-1,-1]
        det += 2*np.log(val)   
        
    # returned values
    res['approx'] = det
    res['sparse'] = sparsity
    res = dotdict(res)
    return(res)
