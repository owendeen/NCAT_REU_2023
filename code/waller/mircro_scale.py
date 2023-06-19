import numpy as np
from scipy.sparse import random, linalg
from itertools import chain, combinations
import copy
from generate_positive_definite_matrix_file import *
from dict_of_diags_file import *




AA = generate_positive_definite_matrix(4)
    #generate positive definite matrix


ss = dict_of_diags(AA)



