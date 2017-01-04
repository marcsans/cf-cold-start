
# coding: utf-8

# In[220]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')

import numpy as np
from matrix_factorization import matrix_factorization
from graph_init import *
from similarity import *
from create_R import *
from ALS import *
from hard_hfs import *
import copy
import matplotlib.pyplot as plt


# In[221]:

def RMSE(ground, predict):
    
    error = 0
    n = 0
    
    for i in range(len(ground)):
        for j in range(len(ground[0])):
            if ground[i,j] != 0:
                error += (ground[i,j] - predict[i,j])**2
                n += 1
                
    return np.sqrt(error/n)

def RMSEvec(ground, predict):
    
    error = 0
    n = 0
    
    for i in range(len(ground)):
        if ground[i] != 0:
            error += (ground[i] - predict[i])**2
            n += 1
                
    return np.sqrt(error/n)

def meanError(ground_truth,new_res):
    return np.mean(abs((new_res - ground_truth)[ground_truth!=0]))


# In[222]:

def dictfromR(R):

    R_dict = {"Users": np.empty([0]), "Movies": np.empty([0]), "Ratings": np.empty([0])}

    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i,j] != 0:
                R_dict["Users"] = np.append(R_dict["Users"],i)
                R_dict["Movies"] = np.append(R_dict["Movies"],j)
                R_dict["Ratings"] = np.append(R_dict["Ratings"],R[i,j])

    return R_dict


# In[223]:

# R = [
#      [5,3,0,1],
#      [4,0,0,1],
#      [1,1,0,5],
#      [1,0,0,4],
#      [0,1,5,4],
#     ]

# R = np.array(R)

R,R_dict = create_R()

print(R_dict)


# In[224]:

num_cold = 5
cold_movies = np.argsort([np.sum(P[i,:] for i in range(len(P)))][0])[-num_cold:]


# In[225]:

to_keep = 3
ground_truth = []
sel = []
for i,c in enumerate(cold_movies):
    sel.append(np.where(R[:,c] != 0)[0])
    np.random.shuffle(sel[i])
    sel[i] = sel[i][:len(sel[i])-to_keep]
    ground_truth.append(copy.deepcopy(R[:,c]))
    R[sel[i],c]=0
    


# In[226]:

R_dictCopy = copy.deepcopy(R_dict)
R_dict = dictfromR(R)


# In[227]:

np.where(R_dict['Movies'] == 321)


# In[228]:

N = len(R)
M = len(R[0])
K = 4

# P = np.random.rand(N,K)
# Q = np.random.rand(M,K)

# nP, nQ = matrix_factorization(R, P, Q, K)

als = ALS(K,N,M,"Users","Movies","Ratings",lbda = 0.1,lbda2 = 0.1)
print("Als created")
ans = als.fit(R_dict)

# nR = np.dot(nP, nQ.T)

# print(nP, "\n\n", nQ)


# In[229]:

R_rec = np.dot(als.U,np.transpose(als.V))


# In[230]:

print(RMSE(R,R_rec))
print(RMSE(R,(R_rec-np.min(R_rec))*5/np.max(R_rec-np.min(R_rec))))


# In[231]:

for i in range(num_cold):
    print(RMSEvec(ground_truth[i], R_rec[:,321]))


# In[232]:

R


# In[233]:

np.max(R_rec)


# In[234]:

lp = LaplacianParams()

# sim = similarity(als.U)
sim = build_graph(als.U, GraphParams())
# Seems to work better with U... 

print(sim)


# In[235]:

L = build_laplacian(sim,lp)

print(L.shape)


# In[236]:

supp = 100
test_vec = copy.deepcopy(R[:,321])*2
# test_vec[:supp] = [0 for i in range(supp)]
test_vec.shape


# In[237]:

# test_vec


# In[238]:

hfs0, confidence = simple_hfs(als.U, test_vec, L, sim)
# hfs0/2


# In[239]:

maxconfidences = np.array([max(confidence[i,:]) for i in range(len(confidence))])


# In[240]:

lim = np.percentile(maxconfidences, 1)
for i in range(num_cold):
#     print(RMSEvec(ground_truth[i]*(maxconfidences > lim),hfs0/2))
    print(RMSEvec(ground_truth[i]*(maxconfidences > lim), R_rec[:,cold_movies[i]]))


# In[241]:

for i in range(num_cold):
#     print(meanError(ground_truth[i],hfs0/2))
    print(meanError(ground_truth[i],R_rec[:,cold_movies[i]]))


# In[242]:

# elmnt = 321
# val = []
# print(RMSEvec(R[:,elmnt],R_rec[:,elmnt]))
# for supp in range(1,671,10):
#     test_vec = copy.deepcopy(R[:,elmnt])*2
#     test_vec[:supp] = [0 for i in range(supp)]

#     hfs0 = simple_hfs(als.U, test_vec, L, sim)
#     val.append(RMSEvec(R[:,elmnt],hfs0/2))
    
# plt.plot(range(1,671,10),val)


# In[243]:

lhfs = []
lconf = []
for i in range(len(R[0])):
    if i%1000 == 0:
        print(i)
    hfs0, confidence = simple_hfs(als.U, R[:,i]*2, L, sim)
    maxconfidences = np.array([max(confidence[i,:]) for i in range(len(confidence))])
    
    lim = np.percentile(maxconfidences, 95)
    
    lhfs.append(hfs0/2)
    lconf.append(maxconfidences > lim)

R_barre = np.vstack(lhfs).T
confs = np.vstack(lconf).T
    
# R_barre[R_barre < 1] = .5
# R_barre[R_barre > 5] = 5


# In[244]:

R_barre_limited = R_barre * confs


# In[245]:

np.unique(R_barre_limited)


# In[246]:

print(RMSE(R_barre_limited,R_rec))


# In[247]:

R_barre_final = copy.deepcopy(R_barre_limited)
R_barre_final[R != 0] = 0


# In[248]:

# R_dict_barre = dictfromR(R_barre_final)


# In[249]:

N = len(R)
M = len(R[0])
K = 4

als_trans = ALS(K,N,M,"Users","Movies","Ratings",lbda = 0.1,lbda2 = 0.1)
print("Als created")

ans = als_trans.fitTransductive(R_dict,R_barre_final,C1=1,C2=0.5)


# In[250]:

R_rec_trans = np.dot(als_trans.U,np.transpose(als_trans.V))
print(RMSE(R_rec_trans,R_rec))


# In[251]:

for i in range(num_cold):
    print("movie "+str(i+1))
    print(RMSEvec(ground_truth[i],R_rec_trans[:,cold_movies[i]]))
    print(RMSEvec(ground_truth[i],R_rec[:,cold_movies[i]]))
    print(RMSEvec(ground_truth[i],R_barre[:,cold_movies[i]]))


# 0.930864602802
# 0.97011494657
# 1.00036650175 (90 1 0.1)
# 
# 0.910201784408
# 0.97011494657
# 1.00036650175 (90 1 0.5)
# 
# 0.931048133417
# 0.97011494657
# 1.00036650175 (95 1 0.5)

# In[135]:

R_barre


# In[ ]:



