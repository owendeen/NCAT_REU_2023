# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:38:53 2023

@author: Colton School
"""

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.linalg import cholesky     #not sure if needed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian


from generate_matrices import laplacian_grid,florida_sparse
import time



def apx_dA_sparse(A , pattern):
    #pattern inputed should not be tril (it can be tho, just redundancy)
    
    #initiliaze lists and some prep
    m,n = A.get_shape()
    pattern = sp.sparse.tril(pattern, format='csr')
    gamma_i_s = np.zeros(n) #these are the L_i_ni,ni s
    
    #now to calculate the gamma_i_s and store them
    for i in range(n):  #i is the ith row
        nonzero_cols = pattern.indices[pattern.indptr[i]:pattern.indptr[i+1]] #in row i, col number of nonzeros
        A_i = (A[ nonzero_cols ,:])[:,nonzero_cols]   #takes rows according to Pi A , then A PiT
                                #may be able to do A_i in one slice, but I don't think so
                                #A_i is square. size = (number of nonzeros in row i of pattern)^2
                                #perhaps would be faster to have a .tocsc() before the second slice, but the
                                #   first slice should make it fairly small. Timing would need to be performed
        
        #these arrays should be relatively small but, might be a sparse version? it would not take csr format
        L_i = sp.linalg.cholesky(A_i.toarray(), lower=True, overwrite_a=False, check_finite=True)

        #where will differ from creating the Ghat
        gamma_i_s[i] = (L_i[-1,-1])**(2/n)
    apx_of_dA = gamma_i_s.prod()
    return(apx_of_dA)



def pure_powerpat_apx(A , apx_to_power, more_than_one_apx = True, save_spspat = False):
    #the first True is whether you want just one apx or all of the ones leading up to it
    #if the save_spat is true, that is for the spline apx
    if more_than_one_apx == False:
        A_pattern = A != 0
        A_pattern = A_pattern**(apx_to_power)
        apx_of_dA = apx_dA_sparse(A , A_pattern)
        return(apx_of_dA)
    
    else:
        m,n = A.get_shape()
        total_entries = m*n
        sparsities = np.zeros(apx_to_power)
        
        
        A1 = A != 0
        tot_nzs = A1.count_nonzero() 
        sparsities[0] = tot_nzs/total_entries
        
        
        apxs = np.zeros(apx_to_power)
        apxs[0] = apx_dA_sparse(A , A1)
        
        A_pattern = A1
        
        for i in range(apx_to_power-1):
            A_pattern = A_pattern @ A1
            
            apxs[i+1] = apx_dA_sparse(A , A_pattern)
            
            tot_nzs = A_pattern.count_nonzero() 
            sparsities[i+1] = tot_nzs/total_entries
        if save_spspat== False:
            return(apxs, sparsities)
        else:
            return(apxs, sparsities, [A1,A_pattern], total_entries)





def graph_spline_apx(A , apx_to_power , apx_to_by_spline): 
    apxs, sparsities, patterns, total_entries = pure_powerpat_apx(A , apx_to_power, more_than_one_apx = True, save_spspat = True)
    
    nu = apx_to_by_spline - apx_to_power  #check how far we need to go
    
    for loop in range(nu):
        patterns[1] = patterns[1] @ patterns[0]
    
    
    tot_nzs = patterns[1].count_nonzero() 
    
    sparsities = list(sparsities)
    sparsities.append(tot_nzs/total_entries)
    
    adj_arr = np.zeros((len(sparsities) , len(sparsities)))
    
    for k in range(len(sparsities)-1):
        weightt = 1/(sparsities[k+1] - sparsities[k]) #distance is difference in sparsity percents. weight is reciprocal of that
        adj_arr[k,k+1] = weightt
        adj_arr[k+1,k] = weightt
        
    lap_arr = laplacian(adj_arr)
    print(lap_arr)
    
    A_ = lap_arr[:,-1]                         #only the last two cols of the laplacian::: (-2,-1) for last two. -1 for last 1
    A_ = np.reshape(A_, (len(A_),1))                #necessary for predicting only last column

    b_ = -np.matmul( lap_arr[:,:-1], apxs)    #all but the last two cols of the laplacian
    b_ = np.reshape(b_, (len(b_),1))                #necessary for predicting only last column

    y = np.linalg.lstsq(A_,b_ ,rcond=-1)
    spline_apx_ = y[0][0][0]
    
    return(spline_apx_ , adj_arr)



#%%
"""
dim_enter = 16
A = laplacian_grid(dim_enter) 
#A = florida_sparse('obstclae.mat')      #error when using eris1176.mat

A1 = A != 0

test1 = apx_dA_sparse(A , A1)


test2  = pure_powerpat_apx(A , 4, more_than_one_apx = False, save_spspat = False)


test3a , test3b = graph_spline_apx(A, 3, 4)

#all is working! :) now for new ones
"""



#%%




def apx_dA_dense(A , pattern):
    #pattern inputed should not be tril (it can be tho, just redundancy)
    
    #initiliaze lists and some prep
    m,n = np.shape(A)
    pattern = np.tril(pattern)
    apx_of_dA = 1 #these are the L_i_ni,ni s
    
    #now to calculate the gamma_i_s and store them
    for i in range(n):  #i is the ith row
        
        A_i = (A[pattern[i] ,:])[:,pattern[i]]
            #may not even need pattern, and can just generate the pattern[i]s as going, but idk if thats faster or not
        
        L_i = sp.linalg.cholesky(A_i, lower=True, overwrite_a=False, check_finite=True)
        
        gamma_i = (L_i[-1,-1])**(2/n)
        apx_of_dA = gamma_i *apx_of_dA
    return(apx_of_dA)


def pure_powerpat_apx_dense(A , apx_to_power, more_than_one_apx = True, save_spspat = False):
    #the first True is whether you want just one apx or all of the ones leading up to it
    #if the save_spat is true, that is for the spline apx
    if more_than_one_apx == False:
        A_pattern = A != 0
        A_pattern = np.linalg.matrix_power(A_pattern , apx_to_power)
        apx_of_dA = apx_dA_dense(A , A_pattern)
        return(apx_of_dA)
    
    else:
        m,n = np.shape(A)
        total_entries = m*n
        sparsities = np.zeros(apx_to_power+1)
        
        
        A1 = A != 0
        tot_nzs = np.count_nonzero(A1)
        sparsities[0] = tot_nzs/total_entries
        
        
        apxs = np.zeros(apx_to_power)
        apxs[0] = apx_dA_dense(A , A1)
        
        A_pattern = A1
        
        for i in range(apx_to_power-1):
            A_pattern = A_pattern @ A1
            
            apxs[i+1] = apx_dA_dense(A , A_pattern)
            
            tot_nzs = np.count_nonzero(A_pattern)
            sparsities[i+1] = tot_nzs/total_entries
        if save_spspat== False:
            return(apxs, sparsities)
        else:
            return(apxs, sparsities, [A1,A_pattern], total_entries)



def graph_spline_apx_dense(A , apx_to_power , apx_to_by_spline): 
    apxs, sparsities, patterns, total_entries = pure_powerpat_apx_dense(A , apx_to_power, more_than_one_apx = True, save_spspat = True)
    
    nu = apx_to_by_spline - apx_to_power  #check how far we need to go
    
    for loop in range(nu):
        patterns[1] = patterns[1] @ patterns[0]
    
    
    tot_nzs = np.count_nonzero( patterns[1])
    
    sparsities = list(sparsities)
    sparsities[-1] = (tot_nzs/total_entries)
    
    adj_arr = np.zeros((len(sparsities) , len(sparsities)))
    
    for k in range(len(sparsities)-1):
        weightt = 1/(sparsities[k+1] - sparsities[k]) #distance is difference in sparsity percents. weight is reciprocal of that
        adj_arr[k,k+1] = weightt
        adj_arr[k+1,k] = weightt
        
    lap_arr = laplacian(adj_arr)
    print(lap_arr)
    
    A_ = lap_arr[:,-1]                         #only the last two cols of the laplacian::: (-2,-1) for last two. -1 for last 1
    A_ = np.reshape(A_, (len(A_),1))                #necessary for predicting only last column

    b_ = -np.matmul( lap_arr[:,:-1], apxs)    #all but the last two cols of the laplacian
    b_ = np.reshape(b_, (len(b_),1))                #necessary for predicting only last column

    y = np.linalg.lstsq(A_,b_ , rcond=-1)
    spline_apx_ = y[0][0][0]
    
    return(spline_apx_ , adj_arr)






#%% testing dense functions
"""
dim_enter = 16
A = laplacian_grid(dim_enter) 
#A = florida_sparse('obstclae.mat')      #error when using eris1176.mat
A = A.toarray()

A1 = A != 0

test1 = apx_of_dA= apx_dA_dense(A , A1)



test2  = pure_powerpat_apx_dense(A , 4, more_than_one_apx = False, save_spspat = False)

test3a , test3b = graph_spline_apx_dense(A, 3, 4)


"""

#%% now to compare!

dim_enter = 25
A = laplacian_grid(dim_enter) 
#A = florida_sparse('obstclae.mat')      #error when using eris1176.mat

t_sparse_pure = time.time()
apx_sparse_pure = pure_powerpat_apx(A , 4, more_than_one_apx = False, save_spspat = False)

t_sparse_pure2 = time.time()
print("apx_sparse_pure", t_sparse_pure2-t_sparse_pure)


t_spline_sparse = time.time()
apx_sparse_spline , one =graph_spline_apx(A, 2, 4)

t_spline_sparse2 = time.time()
print("apx_spline_dense", t_spline_sparse2- t_spline_sparse)





#dense testing

dim_enter = 25
A = laplacian_grid(dim_enter) 
#A = florida_sparse('obstclae.mat')      #error when using eris1176.mat
A = A.toarray()

t_dense_pure = time.time()
apx_dense_pure = pure_powerpat_apx_dense(A , 4, more_than_one_apx = False, save_spspat = False)

t_dense_pure2 = time.time()
print("apx_dense_pure", t_dense_pure2- t_dense_pure)

t_spline_dense = time.time()
apx_dense_spline , two =graph_spline_apx_dense(A, 3, 4)

t_spline_dense2 = time.time()
print("apx_spline_dense", t_spline_dense2- t_spline_dense)



#%%
dim_enter = 250
A = laplacian_grid(dim_enter) 
#A = florida_sparse('obstclae.mat') 
A1 = A != 0

t1 = time.time()
firstest = apx_dA_sparse(A, A1)
t2 = time.time()
print(t2-t1)





dim_enter = 250
A = laplacian_grid(dim_enter) 
#A = florida_sparse('obstclae.mat')      #error when using eris1176.mat
A = A.toarray()
A1 = A != 0

t1 =time.time()
secontest = apx_dA_dense(A, A1)
t2 = time.time()
print(t2-t1)




#%%

A = florida_sparse('apache1.mat')
A1 = A != 0


t1 = time.time()
guess = apx_dA_sparse(A, A1)
t2 = time.time()
print(t2-t1)



