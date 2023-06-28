# -*- coding: utf-8 -*-
"""
this will be the MAIN FILE
"""
#%% Imports
from generate_matrices import laplacian_grid, florida_sparse
from gehat_ge_dghat_old import sparse_apprx_inv
from find_sparsity_pattern_func_old import find_sprs_pttn
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import laplacian

#%% Matrix generation and pre-stuff
dim_enter = 4
A = laplacian_grid(dim_enter)       #generate input^2 x input^2 matrix
m,n = A.get_shape()
orders_of_A = 4             #power of A to go to

A_LU = sp.sparse.linalg.splu(A)     # LU decomp
actual_det = A_LU.L.diagonal().prod() * A_LU.U.diagonal().prod()      #product of the products of the diagonal entries of the LU decomp
actual_value = actual_det**(1 / (dim_enter**2))

#%% A1 and A2 "masks"
A1 = A != 0
sprs_ptts = [A1]
for k in range(2,orders_of_A+1):
    sprs_ptts.append( (sprs_ptts[0])**k )




#%% Ghat and dictionary  REPLACE SECTION AT END

pttns_nondense = []
for k in range(orders_of_A):
    pttns_nondense.append( sprs_ptts[k].todense() )
dict_pttns = []
for k in range(orders_of_A):
    dict_pttns.append( find_sprs_pttn( pttns_nondense[k] ) )


testing123 = []
aprxs = []
for dictt in dict_pttns:
    Ghat , G, dGhat = sparse_apprx_inv(A, dictt)
    aprxs.append(dGhat)
    testing123.append(Ghat)



#%% connectedness and adjacency

#can probably "subtract" the A_nums from one another to get "difference matrices" (or somthn like that)
#and number of values in there will be diff in spsty and give distance.
#then weight. then create adjacency numpy ARRAY and laplacian etc etc.
#hopefully a good bit should be copy-and-paste-able

#assume all A1 subset of A2 etc.

#work on the matrices inbetween
nonsame_entries = []
for k in range(orders_of_A-1):
    nonsame_entries.append( sprs_ptts[k+1] - sprs_ptts[k] )  #entry one in list is A2 - A1

diffs_in_spsty = []     # "distances"
for k in range(len(nonsame_entries)):
    diffs_in_spsty.append(len(nonsame_entries[k].nonzero()[0] )) #number of "grown" entries from A^n to A^(n+1)


adjacency_arr = np.zeros(( orders_of_A , orders_of_A ))
for k in range(len(nonsame_entries)):
    weightt = (1 / diffs_in_spsty[k])       #weight and distance are "normalized" to anything
    adjacency_arr[k,k+1] = weightt
    adjacency_arr[k+1,k] = weightt

lap_arr = laplacian(adjacency_arr)


#%%spline
#going to minimize || Laplacian * [apprxs] ||^(2)_(2)
#to do so going to minimize (last cols of lapl)*(approx we want) = -(other cols of lapl)*(other approxs)
#                                   A                   x        =                     b         

A_ = lap_arr[:,-1]                         #only the last two cols of the laplacian (-2,-1) for last two. -1 for last 1
A_ = np.reshape(A_, (len(A_),1))                #necessary for predicting only last column

b_ = -np.matmul( lap_arr[:,:-1], aprxs[:-1])    #all but the last two cols of the laplacian
b_ = np.reshape(b_, (len(b_),1))                #necessary for predicting only last column

y = np.linalg.lstsq(A_,b_ )
spline_aprx_ = y[0]

#in order to predict these but not actually calculate the last two, take the last 
#two A^nums out of the G calculator

#%% visualization 


round_adj_arr = np.around(adjacency_arr,decimals=3)


graph_arr = nx.from_numpy_array(round_adj_arr, create_using=nx.Graph)
#not sure if we want digraph or not
layout = nx.circular_layout(graph_arr)
#can use different layouts
nx.draw_networkx(graph_arr,layout)
labels = nx.get_edge_attributes(graph_arr, "weight")
nx.draw_networkx_edge_labels(graph_arr, pos=layout, edge_labels=labels)
plt.show()


plt.plot([j for j in range(len(aprxs))],aprxs, 'o')
#plt.plot( [len(aprxs)-2,len(aprxs)-1], spline_aprx_, '^',mfc = 'r')   #for two approxs
plt.plot(len(aprxs)-1, spline_aprx_, '^',mfc = 'r')     #for one approxs
plt.show()



