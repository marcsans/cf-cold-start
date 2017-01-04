
# coding: utf-8

# In[97]:

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


# In[98]:

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


# In[99]:

def dictfromR(R):

    R_dict = {"Users": np.empty([0]), "Movies": np.empty([0]), "Ratings": np.empty([0])}

    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i,j] != 0:
                R_dict["Users"] = np.append(R_dict["Users"],i)
                R_dict["Movies"] = np.append(R_dict["Movies"],j)
                R_dict["Ratings"] = np.append(R_dict["Ratings"],R[i,j])

    return R_dict


# In[100]:

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


# In[101]:

P_dict = copy.deepcopy(R_dict)
P_dict["Ratings"] = np.ones([len(R_dict["Ratings"])])
P = R > 0
print(P)


# In[102]:

np.argmax([np.sum(P[i,:] for i in range(len(P)))][0])


# In[103]:

to_keep = 3
sel = np.where(R[:,321] != 0)[0]
np.random.shuffle(sel)
sel = sel[:len(sel)-to_keep]


# In[104]:

ground_truth = copy.deepcopy(R[:,321])
R[sel,321] = 0


# In[105]:

np.sum(R[:,321] != 0)


# In[106]:

R_dictCopy = copy.deepcopy(R_dict)
R_dict = dictfromR(R)


# In[107]:

np.where(R_dict['Movies'] == 321)


# In[108]:

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


# In[109]:

R_rec = np.dot(als.U,np.transpose(als.V))


# In[110]:

print(RMSE(R,R_rec))
print(RMSE(R,(R_rec-np.min(R_rec))*5/np.max(R_rec-np.min(R_rec))))


# In[111]:

print(RMSEvec(ground_truth, R_rec[:,321]))


# In[112]:

R


# In[113]:

np.max(R_rec)


# In[114]:

lp = LaplacianParams()

# sim = similarity(als.U)
sim = build_graph(als.U, GraphParams())
# Seems to work better with U... 

print(sim)


# In[115]:

L = build_laplacian(sim,lp)

print(L.shape)


# In[116]:

supp = 100
test_vec = copy.deepcopy(R[:,321])*2
# test_vec[:supp] = [0 for i in range(supp)]
test_vec.shape


# In[117]:

# test_vec


# In[118]:

hfs0, confidence = simple_hfs(als.U, test_vec, L, sim)
# hfs0/2


# In[119]:

maxconfidences = np.array([max(confidence[i,:]) for i in range(len(confidence))])


# In[120]:

lim = np.percentile(maxconfidences, 1)
print(RMSEvec(ground_truth*(maxconfidences > lim),hfs0/2))
print(RMSEvec(ground_truth*(maxconfidences > lim), R_rec[:,321]))


# In[121]:

print(meanError(ground_truth,hfs0/2))
print(meanError(ground_truth,R_rec[:,321]))


# In[122]:

# elmnt = 321
# val = []
# print(RMSEvec(R[:,elmnt],R_rec[:,elmnt]))
# for supp in range(1,671,10):
#     test_vec = copy.deepcopy(R[:,elmnt])*2
#     test_vec[:supp] = [0 for i in range(supp)]

#     hfs0 = simple_hfs(als.U, test_vec, L, sim)
#     val.append(RMSEvec(R[:,elmnt],hfs0/2))
    
# plt.plot(range(1,671,10),val)


# In[136]:

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


# In[137]:

R_barre_limited = R_barre * confs


# In[138]:

np.unique(R_barre_limited)


# In[139]:

print(RMSE(R_barre_limited,R_rec))


# In[140]:

R_barre_final = copy.deepcopy(R_barre_limited)
R_barre_final[R != 0] = 0


# In[141]:

# R_dict_barre = dictfromR(R_barre_final)


# In[145]:

N = len(R)
M = len(R[0])
K = 4

als_trans = ALS(K,N,M,"Users","Movies","Ratings",lbda = 0.1,lbda2 = 0.1)
print("Als created")

ans = als_trans.fitTransductive(R_dict,R_barre_final,C1=1,C2=0.2)


# In[146]:

R_rec_trans = np.dot(als_trans.U,np.transpose(als_trans.V))
print(RMSE(R_rec_trans,R_rec))


# In[147]:

print(RMSEvec(ground_truth,R_rec_trans[:,321]))
print(RMSEvec(ground_truth,R_rec[:,321]))
print(RMSEvec(ground_truth,R_barre[:,321]))


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



