
# coding: utf-8

# In[1]:

import numpy as np
from similarity import similarity


# In[2]:

def weightedAverage(neighbors, similarities, R, user, product):
    if neighbors==[]:
        print('no neighbor provided')
    sumWeights = 0
    average = 0
    for n in neighbors:
        if R(n,product)!=0:
            w = similarities(n, user)
            sumWeights = sumWeights + w
            average = average + w * R(n,product)
    average = average / sumWeights
    return average
        


# In[ ]:



