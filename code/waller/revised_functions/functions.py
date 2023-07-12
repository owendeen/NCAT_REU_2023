# -*- coding: utf-8 -*-
"""
file to perform functions:
    apx_dA_no_G
"""
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from scipy.linalg import cholesky     #not sure if needed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import laplacian


from generate_matrices import laplacian_grid,florida_sparse
import time

"""
the apx_dA function is the function that actually takes a sps pattern and approximates dA

it uses slicing in the default false-false state. 

it has two "options". Having build_Ghat=True creates the Ghat_El to be observed for testing
    the second option is csr_Ghat=True, which requires build_Ghat=True. it returns a 
    csr format. it is slower, but handles larger matrices
"""

def apx_dA(A , pattern, build_Ghat=False , csr_Ghat = False ):
    if build_Ghat==False:
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
    
    else:
        if csr_Ghat == False:
        
        
        #pattern inputed should not be tril (it can be tho, just redundancy)
              
          #initiliaze lists and some prep
          m,n = A.get_shape()
          pattern = sp.sparse.tril(pattern, format='csr')
          Ghat_El = np.zeros((n,n)) #these are the L_i_ni,ni s
          
          #now to calculate the gamma_i_s and store them
          for i in range(n):  #i is the ith row
              nonzero_cols = pattern.indices[pattern.indptr[i]:pattern.indptr[i+1]] #in row i, col number of nonzeros
              A_i = (A[ nonzero_cols ,:])[:,nonzero_cols]   #takes rows according to Pi A , then A PiT
                                      #may be able to do A_i in one slice, but I don't think so
                                      #A_i is square. size = (number of nonzeros in row i of pattern)^2
    
              ehat_i = np.zeros(( len(nonzero_cols) ,1))
              ehat_i[-1,0] = 1
              
              ghat_i = sp.linalg.solve(A_i.toarray(), ehat_i )
              Ghat_El[i, nonzero_cols ] = np.transpose(ghat_i)
              
          apx_of_dA = np.prod(np.diagonal(Ghat_El)**(-1/n))
          
          return(apx_of_dA, Ghat_El)
          
        else:
          m,n = A.get_shape()
          pattern = sp.sparse.tril(pattern, format='csr')
          Ghat_El_rows = list(np.zeros(n)) #these are the L_i_ni,ni s
          
          #now to calculate the gamma_i_s and store them
          for i in range(n):  #i is the ith row
              nonzero_cols = pattern.indices[pattern.indptr[i]:pattern.indptr[i+1]] #in row i, col number of nonzeros
              A_i = (A[ nonzero_cols ,:])[:,nonzero_cols]   #takes rows according to Pi A , then A PiT
                                      #may be able to do A_i in one slice, but I don't think so
                                      #A_i is square. size = (number of nonzeros in row i of pattern)^2
    
              ehat_i = np.zeros(( len(nonzero_cols) ,1))
              ehat_i[-1,0] = 1
              
              ghat_i = sp.linalg.solve(A_i.toarray(), ehat_i )
              
              g_i = np.zeros(n)
              g_i[nonzero_cols]= np.transpose(ghat_i)
              g_i = sp.sparse.csr_array(g_i)
              
              Ghat_El_rows[i] = g_i
              
          Ghat_El = sp.sparse.vstack(Ghat_El_rows).tocsr()
          
          apx_of_dA = np.prod(sp.sparse.csr_matrix.diagonal(Ghat_El) **(-1/n))
          
          return(apx_of_dA, Ghat_El)
          
          

"""
this function calculates the appxs AND the sparsity patterns

in the default state of True-false-false-false, it calculates the approximation             ***
    at EACH power before the final

more_than_one_apx = False ONLY approximates at the desired sparsity pattern to the power    ***
    this would be used to compare actual times 

with save_spspat = True (and the others being default) , it returns sparsity patterns to be used in the SPLINE FUNCTION

with build_Ghat=True (and others default) creates the Ghat_El to be observed for testing
with csr_Ghat=True, which requires build_Ghat=True. it returns in csr format. it is slower, but handles larger matrices

"""


def pure_powerpat_apx(A , apx_to_power, more_than_one_apx = True, save_spspat = False, build_Ghat=False , csr_Ghat = False):
    #apx_to_power is the power you want apx to
    #more_than_one_apx = False is when you want to do a single appx to a power (fastest)
    #save_final_spspat = False is only for inputting into spline function
    #build Ghat is to get the Ghats
    #csr_ghat is to return Ghat in csr formats, should be slightly faster. its in nparrays if False
    if more_than_one_apx == False:
        A_pattern = A != 0
        A_pattern = A_pattern**(apx_to_power)
        apx_of_dA = apx_dA(A , A_pattern, build_Ghat=False , csr_Ghat = False)
        return(apx_of_dA)
    
    else:
        if build_Ghat== False:
            m,n = A.get_shape()
            total_entries = m*n
            sparsities = np.zeros(apx_to_power)
            
            
            A1 = A != 0
            tot_nzs = A1.count_nonzero() 
            sparsities[0] = tot_nzs/total_entries
            
            
            apxs = np.zeros(apx_to_power)
            apxs[0] = apx_dA(A , A1, build_Ghat=False , csr_Ghat = False)
            
            A_pattern = A1
            
            for i in range(apx_to_power-1):
                A_pattern = A_pattern @ A1
                
                apxs[i+1] = apx_dA(A , A_pattern , build_Ghat=False , csr_Ghat = False)
                
                tot_nzs = A_pattern.count_nonzero() 
                sparsities[i+1] = tot_nzs/total_entries
            if save_spspat== False:
                return(apxs, sparsities)
            else:
                return(apxs, sparsities, [A1,A_pattern], total_entries)
        
        else:
            if csr_Ghat== False:
                m,n = A.get_shape()
                total_entries = m*n
                sparsities = np.zeros(apx_to_power)
                A1 = A != 0
                tot_nzs = A1.count_nonzero() 
                
                
                sparsities[0] = tot_nzs/total_entries
                
                apxs = np.zeros(apx_to_power)
                list_of_Ghats = list(np.zeros(apx_to_power))
                
                apxs[0] , list_of_Ghats[0] = apx_dA(A , A1, build_Ghat=True , csr_Ghat = False)
                
                A_pattern = A1
                
                for i in range(apx_to_power-1):
                    A_pattern = A_pattern @ A1
                    apxs[i+1] , list_of_Ghats[i+1] = apx_dA(A , A_pattern , build_Ghat=True , csr_Ghat = False)
                    
                    tot_nzs = A_pattern.count_nonzero() 
                    sparsities[i+1] = tot_nzs/total_entries
                return(apxs , sparsities, list_of_Ghats)
            
            else:
                m,n = A.get_shape()
                total_entries = m*n
                sparsities = np.zeros(apx_to_power)
                
                A1 = A != 0
                tot_nzs = A1.count_nonzero() 
                sparsities[0] = tot_nzs/total_entries
                
                apxs = np.zeros(apx_to_power)
                list_of_Ghats = list(np.zeros(apx_to_power))
                apxs[0] , list_of_Ghats[0] = apx_dA(A , A1, build_Ghat=True , csr_Ghat = True)
                
                A_pattern = A1
                
                for i in range(apx_to_power-1):
                    A_pattern = A_pattern @ A1
                    apxs[i+1] , list_of_Ghats[i+1] = apx_dA(A , A_pattern , build_Ghat=True , csr_Ghat = True)
                    
                    tot_nzs = A_pattern.count_nonzero() 
                    sparsities[i+1] = tot_nzs/total_entries
                return(apxs , sparsities, list_of_Ghats)
            
            
"""
this function approximates dA to a given sparsity pattern power with PAPER method
then it apxs to a further out sparsity pattern power with spline method

considering being able to apx to a particular sparsity percent instead of just to a sparsity power. sparsity power 
allows for easy comparison. Itd be an easy option to switch between the two. youd start at the sparsities.append line
and just add your given sparsity. maybe it would let you know the sparsities it has stored and prompts you
for one afterwards. i propose this bc we may be losing some time with the calculation of the sparsity pattern and then taking
the sparsity value of that, but I doubt it is really adding much time in at all
"""

def graph_spline_apx(A , apx_to_power , apx_to_by_spline): 
    apxs, sparsities, patterns, total_entries = pure_powerpat_apx(A , apx_to_power, more_than_one_apx = True, save_spspat = True, build_Ghat=False , csr_Ghat = False)
    
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

    y = np.linalg.lstsq(A_,b_ )
    spline_apx_ = y[0][0]
    
    return(spline_apx_ , adj_arr)
        


#now to do visualization ones. idk if I'll make those into a function, bc it'll be quite specific to what
#specific thing we're trying to demonstrate

#%% test apx_of_dA


dim_enter = 16
A = laplacian_grid(dim_enter) 
#A = florida_sparse('obstclae.mat')      #error when using eris1176.mat

A1 = A != 0

apx_of_dA= apx_dA(A , A1,  build_Ghat=False, csr_Ghat = False)


#%% test pure_powerpat_apx
dim_enter = 100
A = laplacian_grid(dim_enter) 

apxs, sparsities  =  pure_powerpat_apx(A , 4, more_than_one_apx = True, save_spspat = False, build_Ghat=False , csr_Ghat = False)
#this is the default. It will calculate the apxs for each power and give the sparsity values


apxs, sparsities, patterns, total_entries = pure_powerpat_apx(A , 4, more_than_one_apx = True, save_spspat = True, build_Ghat=False , csr_Ghat = False)
#this is the one specifically for use in the spline approximation, which is why it has extra outputs
#       it saves from a couple of re-calculations. Might not have really mattered, but I thought, "eh, why not?"


apx_of_dA =  pure_powerpat_apx(A , 4, more_than_one_apx = False, save_spspat = False, build_Ghat=False , csr_Ghat = False)
#this is the speediest one that only calculates one value at one given sparsity pattern at a power


apxs , sparsities, list_of_Ghats = pure_powerpat_apx(A , 4, more_than_one_apx = True, save_spspat = False, build_Ghat=True , csr_Ghat = False)
#this one gives Ghats in nparray form. accessible but might error/inf for anything large/not very sparse


apxs , sparsities, list_of_Ghats = pure_powerpat_apx(A , 4, more_than_one_apx = True, save_spspat = False, build_Ghat=True , csr_Ghat = True)
#this one gives Ghats in csr form. should be able to handle larger A's than nparray form but slower


#%% test spline calculations
dim_enter = 100
A = laplacian_grid(dim_enter) 

spline_apx_ , adj_arr , notspline_apxs ,sparsities = graph_spline_apx(A , 2, 4)




#%% actual value to compare to
#dim_enter = 500
#A = laplacian_grid(dim_enter) 

#%%

t0 = time.time()

m,n = A.get_shape()
A_LU = sp.sparse.linalg.splu(A)
actual_value_calculated = (A_LU.U.diagonal() **(1/(n))  ).prod()

t1 = time.time()
print(t1-t0)


ts0 = time.time()
spline_apx_ , adj_arr = graph_spline_apx(A , 2, 4)
ts1 = time.time()
print(ts1-ts0)

tp0 = time.time()
speedy=  pure_powerpat_apx(A , 4, more_than_one_apx = False, save_spspat = False, build_Ghat=False , csr_Ghat = False)
tp1 = time.time()
print(tp1-tp0)




#need to try with parallelization I think.