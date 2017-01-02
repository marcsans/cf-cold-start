
# coding: utf-8

# In[24]:

def create_R(ratingsstr = "../data/ml-latest-small/ratings.csv"):
    
    import numpy as np
    import pandas as pd
    
    ratings = pd.read_csv(ratingsstr)
    
    uniqueRatings = np.unique(ratings['movieId'])
    ratings['TrueMovieId'] = ratings['movieId'].map(lambda i: np.argmin(abs(uniqueRatings - i)))
    R = np.zeros([len(np.unique(ratings['userId'])),len(uniqueRatings)])
    R_dict = {"Users": np.empty([0]), "Movies": np.empty([0]), "Ratings": np.empty([0])}
    
    ratingsnp = np.asarray(ratings)
    
    for i in range(len(ratings)):
        R[ratingsnp[i,0]-1, ratingsnp[i,-1]] = ratingsnp[i,2]
        R_dict["Users"] = np.append(R_dict["Users"],ratingsnp[i,0]-1)
        R_dict["Movies"] = np.append(R_dict["Movies"],ratingsnp[i,-1])
        R_dict["Ratings"] = np.append(R_dict["Ratings"],ratingsnp[i,2])
    
    return R, R_dict

