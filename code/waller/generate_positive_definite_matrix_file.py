import numpy as np




def generate_positive_definite_matrix(n):
    A = np.random.randint(1, 10*n, size=(n, n))  # Generate random integers between 1 and 10
    A = np.dot(A, A.transpose())              # Multiply A with its transpose
    L = np.linalg.cholesky(A)                 # Cholesky factorization
    return np.dot(L, L.T)



