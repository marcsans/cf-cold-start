from sklearn.utils import as_float_array
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def sparse_matrix_input(X):
    X = as_float_array(X)[:,:3]
    return X

def sparse_matrix(X,n,p,w=1,names=['row','col','val']):
    #X = sparse_matrix_input(X)
    R = csr_matrix((w*X[names[2]], (X[names[0]],X[names[1]])), shape=(n,p))
    return R

def RMSE(R_true,R_pred,W=None):
    if W is None:
        W = R_true.nonzero()
    if 'sparse' in str(type(R_true)):
        return mean_squared_error(np.array(R_true[W])[0],R_pred[W])**.5
    else:
        return mean_squared_error(R_true[W],R_pred[W])**.5

def sparseMatrix(data,k,n,p,include=False,names=['user','item','rating']):
    # if include = True we take only cv=k, ortherwise we only exclude cv=k
    if include:
        R = csr_matrix((data[names[2]][data['cv']==k], 
                                (data[names[0]][data['cv']==k], 
                                 data[names[1]][data['cv']==k])),
                         shape=(n,p))
    else:
        R = csr_matrix((data[names[2]][data['cv']!=k], 
                                (data[names[0]][data['cv']!=k], 
                                 data[names[1]][data['cv']!=k])),
                         shape=(n,p))
    return R

# to delete
def getLine(perf,model,param,cv):
    try: # if the line exists
        out = perf[(perf['model']==model) & (perf['params']==param) & 
            (perf['crossval']==cv)].index[0]
    except IndexError: # create a new line
        try: # the dataset is not empty
            out = max(perf.index)+1
        except ValueError:
            out = 0
    return out
    
def getLine_fromdict(perf,Dict):
    tmp = pd.DataFrame(columns=list(Dict.keys()))
    tmp.loc[0,list(Dict.keys())] = list(Dict.values())
    tmp = pd.merge(perf,tmp)
    try: # the dataset is not empty
        out = max(perf.index)+1
    except ValueError:
        out = 0
    return out

def extract_year(x):
    try:
        return int(x.split('-')[2])
    except AttributeError:
        return -1

def argmax(x):
    p = np.argwhere(x==np.max(x))
    return p[np.random.randint(len(p))][0]

def argmin(x):
    p = np.argwhere(x==np.min(x))
    return p[np.random.randint(len(p))][0]