
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd


# In[ ]:

def similarity(U):
    num_users = U.shape[0]
    dist = np.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(i):
            dist[i,j] = np.abs(np.dot(U[i,:], U[j,:].T))/(np.linalg.norm(U[i,:])*np.linalg.norm(U[j,:]))
            dist[j,i] = np.abs(np.dot(U[i,:], U[j,:].T))/(np.linalg.norm(U[i,:])*np.linalg.norm(U[j,:]))
    return dist

