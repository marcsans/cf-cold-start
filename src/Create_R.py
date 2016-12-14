
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd


# In[6]:

movies = pd.read_csv("../data/ml-latest-small/movies.csv")
ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")


# In[8]:

movies.head()


# In[36]:

ratings.head()


# In[37]:

np.asarray(ratings.head())[:,0]


# In[35]:

R = np.zeros([len(np.unique(ratings['userId'])),len(movies)])


# In[38]:

ratingsnp = np.asarray(ratings)


# In[41]:

for i in range(len(ratings)):
    R[ratingsnp[i,0], ratingsnp[i,1]] = ratingsnp[i,2]


# In[ ]:



