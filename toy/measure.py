# import scipy as sp
import numpy as np

import pandas as pd

from numpy import linalg

import pdb

import matplotlib.pyplot as plt

from scipy import stats
import sklearn as sk
from sklearn import svm
from tqdm import tqdm

def plot(fn,numits=1000,xlabels=["sample value"],plotnames=None,savedata=None):
    N = numits
    rawdata = []
    for it in range(N):
        res = fn()
        rawdata += [res]
        npdata = np.array(rawdata)
        mu = npdata.mean(axis=0)
        sigma = npdata.std(ddof=1,axis=0) if it>0 else 0
        sigma /= np.sqrt(it+1)
        t = sp.stats.t(it)
        p = t.cdf(-mu/sigma) if it>0 else 1
        print("it: ",it,"/",N-1,
              ", mean: ", mu,
              ", std: ", sigma,
              ", p value: ",p,end="\r",flush=True)
    if savedata is not None:
        with open("data/"+str(savedata)+".pk","wb") as f:
            pk.dump(rawdata,f)
        print("saved data to ",savedata)
    
    print("\n")

    lo = mu+(t.ppf(0.05)*sigma)
    for ind,lab in enumerate(xlabels):
        # lo = mu+(t.ppf(0.025)*sigma)
        # hi = mu+(t.ppf(0.975)*sigma)
        xs = np.arange(N)
        ones = np.ones(N)
        _=plt.hist(npdata[:,ind],bins=max(int(N/10),1))
        plt.ylabel("num of samples")
        plt.xlabel("sample value of "+lab)
        plt.axvline(x=lo[ind],color="r",label="95%% confidence lower bound for sample mean (%f)"%lo[ind])
        plt.legend()
        plt.title("histogram of "+lab)
        # plt.plot(xs,data,'rs',xs,ones*lo,'k-',xs,np.zeros(xs.shape),'w-')
        # plt.plot(xs,data,'rs',xs,ones*lo,'g--',xs,ones*hi,'g--')
        if plotnames is None: plt.show()
        else: plt.savefig("figs/"+plotnames[ind]+".jpg")
        plt.clf()