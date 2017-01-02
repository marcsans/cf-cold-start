
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

def similarity(U):
    num_users = U.shape[0]
    sim = np.zeros((num_users, num_users))
    min_dist = 0.001
    for i in range(num_users):
        for j in range(i):
            s = 1 / (min_dist + np.arccos(np.dot(U[i,:], U[j,:].T))/(np.linalg.norm(U[i,:])*np.linalg.norm(U[j,:])) / np.pi)
            sim[i,j] = s
            sim[j,i] = s
    return sim


# In[ ]:



