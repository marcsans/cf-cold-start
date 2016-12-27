
# coding: utf-8

# In[24]:

def create_R(moviesstr = "../data/ml-latest-small/movies.csv", ratingsstr = "../data/ml-latest-small/ratings.csv"):
    
    import numpy as np
    import pandas as pd
    
    movies = pd.read_csv(moviesstr)
    ratings = pd.read_csv(ratingsstr)
    
    ratings['TrueMovieId'] = ratings['movieId'].map(lambda i: movies[movies.movieId == i].index.tolist()[0])
    R = np.zeros([len(np.unique(ratings['userId'])),len(movies)])
    
    ratingsnp = np.asarray(ratings)
    
    for i in range(len(ratings)):
        R[ratingsnp[i,0]-1, ratingsnp[i,-1]] = ratingsnp[i,2]
    
    return R

