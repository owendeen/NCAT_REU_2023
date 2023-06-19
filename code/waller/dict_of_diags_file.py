import numpy as np
import copy




def dict_of_diags(A):
  #this makes a list of all
    m,n = np.shape(A)
    diagonals = []

    ss= []



    for k in range(1,m+1):
        base_dict = {}
        for kk in range(1,m+2-k):
          base_dict[kk-1+k] = [kk]
        diagonals.append(base_dict)

    ss.append(diagonals[0])

    for r in range(len(diagonals)-1):
        new_dict = copy.deepcopy( ss[r] )
        for rr in range(r+2 ,len(diagonals)+1):
           new_dict[rr] = diagonals[r][rr-1]  + new_dict[rr]
        ss.append(new_dict)


    return ss
