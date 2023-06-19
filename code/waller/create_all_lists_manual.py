import numpy as np
from itertools import chain, combinations
import copy
import csv



def powset(iterable):
  #input a list or range
  nums = list(iterable)
  return list(chain.from_iterable(combinations(nums,r) for r in range(len(nums)+1)))

#%%
a = np.ones((4,4))
m,n = np.shape(a)



#create a dictionary with the key for every row and the diagonal
base_dict = {}
for cc in range(1,m+1):
  base_dict[cc] = [cc]

    #^delete print line at end

#%%

rows_c =[]
#now to create all "rows"
for i in range(1,m):
    roww = powset(range(1,i+1))
    rows_c.append( [list(i) for i in roww] )


#%%
tot = 2**( (m*(m -1))/2)
#number of dictionaries


#%%

#list of dictionaries

ss=[]
dicct = copy.deepcopy(base_dict)
i=0
while i < tot:
    for q in range(len(rows_c[0])):

        for w in range(len(rows_c[1])):

            for r in range(len(rows_c[2])):
                dicct[2] = rows_c[0][q]  + dicct[2]
                dicct[3] = rows_c[1][w] + dicct[3]
                dicct[4] = rows_c[2][r]  + dicct[4]
                ss.append(dicct)
                dicct = copy.deepcopy(base_dict)
                i += 1














