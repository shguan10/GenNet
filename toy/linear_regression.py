import numpy as np
import scipy as sp
import pandas as pd
from numpy import linalg
from scipy import stats
import pdb
import pickle as pk
import matplotlib.pyplot as plt

from utils import *

BENEFIT = 0
DIFFNORM = 1
BIAS = 2
NORMDIFF = 3
DIFFDOTREAL = 4
DIFFTHEORY = 5

def linear_reg_exp(numtrain=3000,numtest=300,m1weight=(1,),m2weight=(1,),metric=BENEFIT,m2bias=[0],useM2avg=True,showBeta=False,translate=False,corrN=[0,0],frobNorm=1,partialTranslate=False):
  """
  The correlation model we are using is a very simple one. Specify the tuple (how many from m1, how many from m2) will be correlated. This will specify the dimensions of the translation matrix from m1 and m2. 

  frobNorm will be the parameter specifying the Frobenius norm of the 'cross correlation matrix' between m1 and m2, which we'll define as E[m1 m2^T] - E[m1]E[m2^T] = E[m1 m2^T]. This is the same as the Frobenius norm of the translation matrix, which is
  Z where m_2 = Z @ m_1 + epsilon.
  """

  numdata = numtrain + numtest
  C1 = len(m1weight)
  C2 = len(m2weight)
  numfeat = 1+C1+C2
  m2start = 1+C1

  bias = np.ones((numdata,1)).astype(np.float64)
  realm1 = np.random.normal(size=(numdata,C1)).astype(np.float64)
  realm2 = np.random.normal(size=(numdata,C2)).astype(np.float64)
  corrpresent = corrN[1] > 0

  if corrpresent:
    transmatrix = np.random.normal(size=corrN)
    transmatrix *= frobNorm / np.linalg.norm(transmatrix)

    # now generate m2 from m1
    addm2 = realm1[:,:corrN[0]] @ transmatrix

    realm2[:,:corrN[1]] += addm2

  xs = np.concatenate((bias,realm1,realm2),axis=1)
  m2bias = np.array(m2bias).mean()
  realbeta = [0]+list(m1weight) + list(m2weight)
  realbeta = np.array(realbeta).reshape(-1,1).astype(np.float64)

  y = xs @ realbeta
  y += np.random.normal(size=(y.shape))

  # do the first model
  X = xs[:numtrain,:m2start]
  Y = y[:numtrain,:]
  Xt = np.transpose(X)
  XtX = Xt @ X
  XtXinv = np.linalg.inv(XtX)
  hatbeta = XtXinv @ Xt @ Y

  testX = xs[numtrain:,:m2start]
  prederr = (testX @ hatbeta) - y[numtrain:,:]
  uprederr = (np.transpose(prederr) @ prederr).squeeze()

  # do the second model
  X = xs[:numtrain,:]
  Y = y[:numtrain,:]
  if translate:
    if not partialTranslate:
      M1 = X[:,:m2start]
      M2 = X[:,m2start:]
    else:
      M1 = realm1[:,:corrN[0]]
      M2 = realm2[:,:corrN[1]]

    M1t = np.transpose(M1)
    M1tM1 = M1t @ M1
    M1tM1inv = np.linalg.inv(M1tM1)
    hatbetatrans = M1tM1inv @ M1t @ M2

  Xt = np.transpose(X)
  XtX = Xt @ X
  XtXinv = np.linalg.inv(XtX)
  hatbeta2 = XtXinv @ Xt @ Y

  if showBeta:
    plt.plot(hatbeta,'g',label="control hatbeta")
    plt.plot(hatbeta2,'r',label="mm hatbeta")
    plt.plot(realbeta,"b--",label="real beta")
    plt.xlabels("index of (hat)beta")
    plt.legend()
    plt.title("hatbeta for M2 ~ N(%.1f,1)"%m2bias)
    plt.show()
    pdb.set_trace()

  m1test = xs[numtrain:,:m2start]
  if translate:
    if not partialTranslate:
      m2test = m1test @ hatbetatrans
    else:
      m2test = np.ones((numtest,numfeat-m2start))*X[:numtrain,m2start:].mean(axis=0)
      m2test[:,:corrN[1]] = m1test[:,:corrN[0]] @ hatbetatrans
  else:
    m2test = np.ones((numtest,numfeat-m2start))*X[:numtrain,m2start:].mean(axis=0)
    m2test*= useM2avg
  testX = np.concatenate((m1test, m2test),axis=1)
  prederr = (testX @ hatbeta2) - y[numtrain:,:]
  bprederr = (np.transpose(prederr) @ prederr).squeeze()
  if metric==BENEFIT: 
    # M1 = xs[:numtrain,:-1]
    # M1p= xs[numtrain:,:-1]
    # M2 = xs[:numtrain,-1].reshape(-1,1)
    # M2p= xs[numtrain:,-1].reshape(-1,1)
    # M1t = M1.transpose()
    # B = M1p @ np.linalg.inv(M1t @ M1)@ M1t @ M2
    # v = (B**2).sum()
    
    # V = M1p @ np.linalg.inv(M1t @ M1)@ M1t
    # v = (V.transpose() @ V).trace()

    # H = M1 @ np.linalg.inv(M1t@M1) @ M1t
    # v = H.trace()
    # v = (M2p*l).sum()
    # v = (l**2).sum()-2*(M2p*l).sum()
    # v = ((M2p - l)**2).sum()
    # l = (l**2).sum()
    # r = 4 * (M2p**2).sum()
    return uprederr - bprederr
  elif metric==BIAS: return (hatbeta[0]-hatbeta2[0]).squeeze()
  elif metric==DIFFNORM: return np.sum(hatbeta*hatbeta).squeeze()-np.sum(hatbeta2[:-1]*hatbeta2[:-1]).squeeze()
  elif metric==NORMDIFF: return np.sum((hatbeta-hatbeta2[:-1])**2).squeeze()
  elif metric==DIFFDOTREAL: return np.sum((hatbeta2[:-1]-hatbeta)*realbeta[:-1]).squeeze()
  elif metric==DIFFTHEORY:
    exp = uprederr - bprederr
    M1=  xs[:numtrain,:m2start]
    M1p = xs[numtrain:,:m2start]
    theory = (realbeta[m2start:]**2).sum()*(C1+1)*numtest / (numtrain-C1-2)
    return theory - exp

if __name__ == '__main__':
  metric = BENEFIT
  if metric==BENEFIT: xlabels=["mm benefit"]
  elif metric==DIFFNORM:  xlabels=["diff in norm (control - mm)"]
  elif metric==NORMDIFF:  xlabels=["norm of the diff in hatbeta of M1"]
  elif metric==BIAS: xlabels=["diff in hatbeta[0] (control - mm)"]
  elif metric==DIFFDOTREAL: xlabels=["diff in hatbeta dot product with realbeta (mm - control)"]
  elif metric==DIFFTHEORY: xlabels=["diff in mmbenefit (theory - actual)"]

  # diffvar=False
  # xlabels = ["mm benefit","testloss1","difftestvar"] if diffvar else ["mm benefit","testloss1"]

  data = []

  for r1 in [20,40,60,80,100]:
    for r2 in [0,1,2,4,8,16,32,64,128]:
    # for c2 in [20,40,60,80,100]:

      for frobNorm in np.linspace(0,2,num=10,endpoint=False):

        numits=1000
        numtrain=1000
        numtest=1000
        c1 = 100
        c2 = 128
        translate = True
        partialTranslate=True

        # corr2 = c2
        # corrN=[c1,corr2]
        corrN=[r1,r2]

        show = False
        def fn(show=show):
          return [linear_reg_exp(numtrain=numtrain,numtest=numtest,m1weight=[1]*c1,m2weight=[1]*c2,m2bias=[0],useM2avg=True,showBeta=False,metric=metric,translate=translate,corrN=corrN,frobNorm=frobNorm,partialTranslate=partialTranslate)]
        # fn(True)
        name = str(r1)+"_"+str(r2)+"_"+str(frobNorm)+"_"+"partial"
        # plotnames=[name+"benefit",name+"loss",name+"diffval"]
        plotnames=[name+"benefit"]
        savedata=name
        show=True

        print(name)
        data.append((name,getdata(fn,numits=numits,savedata=savedata)))
    
  with open("data/linearcorr_allcorr_partial.pk","wb") as f:
    pk.dump(data,f)