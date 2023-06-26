import networkx as nx
import numpy as np
import scipy as sp


def laplacian_grid(n):
    N = n+2 
    G = nx.grid_graph(dim=(range(N), range(N)))  # generate graph
    Lap = nx.laplacian_matrix(G)                 # find laplacian
    mask = Lap.diagonal() == 4                   # find interior points
    Lcut = Lap[mask,:][:,mask]                   # truncate laplacian to interior points
    return(Lcut)


def florida_sparse(fname):
    if fname in ['eris1176.mat']:
        A = list(sp.io.loadmat(fname)['Problem'][0,0])
        X = A[1]
    elif fname in ['apache1.mat','obstclae.mat']:
        A = list(sp.io.loadmat(fname)['Problem'][0,0])
        X = A[2]
    return(X)




if __name__ == '__main__':
    A = laplacian_grid(4)
    B = florida_sparse('apache1.mat')
    C = florida_sparse('eris1176.mat')
    D = florida_sparse('obstclae.mat')
