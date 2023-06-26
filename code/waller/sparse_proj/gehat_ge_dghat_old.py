import numpy as np


#these are functions to get the G and Ghat. They are good, but I think they
#can be redone such that only the "diagonal" values of a particular matrix need to 
# be calculated etc. etc. 



#creates basis vector, from the identity matrix, of appropriate sizes
def get_basis_vector(i, n):
  '''
  Helper function
  '''

  identity_matrix = np.eye(n)
  basis_vector = identity_matrix[:, i]
  return basis_vector


def sparse_apprx_inv(A, pattern):
  '''
  A is a positive definite symmetric matrix of dim n x n
  pattern is the sparsity pattern stored as a dictionary object
  '''
  m, n = np.shape(A)
  if m == n:
    Pi_s = []
    G_E_hat = np.zeros((n, n))
    for key, value in pattern.items():
      # i num is key
      # value is an arr with the length being the num of rows
      p = np.zeros((len(value), n))
      indc = 0
      for num in value:
        p[indc][num - 1] = 1

        indc += 1
      Pi_s.append(p)

    i = 0
    for p in Pi_s:
      basis_vector = get_basis_vector(i, n)
      g = p.T @ np.linalg.inv((p @ A @ p.T)) @  (p @ basis_vector)
      G_E_hat[i] = g.T
      i += 1


    G_E = (np.diag(G_E_hat) ** -0.5 ) @ G_E_hat # not sure if this correct

    dGhat = (np.linalg.det(G_E_hat) ** (1/n) ) ** (-1)
    #this is the approx of the function

    return G_E_hat, G_E, dGhat


