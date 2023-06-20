import numpy as np
from scipy.sparse import random, linalg
from itertools import chain, combinations
import copy
from generate_positive_definite_matrix_file import *
from dict_of_diags_file import *
from gehat_ge_dghat import *
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
import networkx as nx


#%%     generate positive definite matrix
A = generate_positive_definite_matrix(20)
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

'''
so this first section in here is incomplete, but can be done. might be a little 
tricky with our dictionary approach, but certainly possible. the next takes 
ALL adjacent patterns but idk if we want them to be a linear graph or something (only one connected to another)

#%%     collect the index pairs of dicts that are subsets of another manually
#we know that each sparsity pattern is less sparse than its preceeding one
#and can only be contained in the ones after it




    #this may not work in python versions before 3.9
'''


#%%     collect the index pairs of dicts that are subsets of another by knowing pattern
"""
list_of_adjacents = []
nuu =1
for k in reversed(range(1,len(ss))):
    for kk in range(len(ss)-nuu):
        list_of_adjacents.append((k,kk))
    nuu+=1
"""

list_of_adjacents = []

for k in reversed(range(1,len(ss))):
    list_of_adjacents.append((k,k-1))


#%%     make empty adjacency matrix
adjacency_arr = np.zeros((n,n))

#with n number of dictionaries


#%%     with the index pairs, find difference in their sparsity as weight and store in the ajaceny matrix

for k in range(len(list_of_adjacents)):
    lef, rig = list_of_adjacents[k]
    weightt  = 1/( (entries_in_patterns[lef] - entries_in_patterns[rig]) )
    #weight is 1\distance with distance being difference in number of entries. we can
    #normalize the distance by dividing it by n or (n/2) or something
    adjacency_arr[lef,rig] = weightt
    adjacency_arr[rig,lef] = weightt




#%%     calculate laplacian
lap_arr = laplacian(adjacency_arr)


#%%     visualize the adjacency matrix to a graph

round_adj_arr = np.around(adjacency_arr,decimals=3)


graph_arr = nx.from_numpy_array(round_adj_arr, create_using=nx.Graph)
#not sure if we want digraph or not
layout = nx.circular_layout(graph_arr)
#can use different layouts
nx.draw_networkx(graph_arr,layout)
labels = nx.get_edge_attributes(graph_arr, "weight")
nx.draw_networkx_edge_labels(graph_arr, pos=layout, edge_labels=labels)
plt.show()


#%% minimize the spline

#the full sparsity pattern gives the actual answer, so we are going to try and use
#all but the last (full) sparsity pattern

#going to minimize || Laplacian * [apprxs] ||^(2)_(2)
#to do so going to minimize (last col of lapl)*(approx we want) = -(other cols of lapl)*(other approxs)
#                                   A                   x        =                     b

A_ = lap_arr[:,-1]
A_ = np.reshape(A_, (len(A_),1))

b_ = -np.matmul( lap_arr[:,:-1], aprxs[:-1])
b_ = np.reshape(b_, (len(b_),1))


y = np.linalg.lstsq(A_,b_ )

# i really don't know what is happening here. I thought i did, but idk. 
#idk why it isn't giving back a singular value, but the first one is a reasonable approximation

spline_aprx_ = y[0]





#%%

#predicting the last two based on the others


A_ = lap_arr[:,(-2,-1)]
A_ = np.reshape(A_, (len(A_),2))

b_ = -np.matmul( lap_arr[:,:-2], aprxs[:-2])
b_ = np.reshape(b_, (len(b_),1))


y = np.linalg.lstsq(A_,b_ )

# i really don't know what is happening here. I thought i did, but idk. 
#idk why it isn't giving back a singular value, but the first one is a reasonable approximation

spline_aprx_ = y[0]






