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

NAIVE="naive"
FULLTRANS="full"
PARTIALTRANS="partial"
ALL="all"

def linear_reg_exp(numtrain=3000,numtest=300,m1weight=(1,),m2weight=(1,),m2bias=[0],useM2avg=True,showBeta=False,corrN=[0,0],frobNorm=1,bimodal=NAIVE,pN=None):
  """
  The correlation model we are using is a very simple one. Specify the tuple (how many from m1, how many from m2) will be correlated. This will specify the dimensions of the translation matrix from m1 and m2. 

  frobNorm will be the parameter specifying the Frobenius norm of the 'cross correlation matrix' between m1 and m2, which we'll define as E[m1 m2^T] - E[m1]E[m2^T] = E[m1 m2^T]. Supposing m2 = Z@m1, then E[m1 m2^T] = E[m1 @ m1^T @ Z]. Under assumption that Var(m1)=I, then the cross correlation matrix is the same as the translation matrix
  """
  if pN is None: pN = corrN

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

  # realm2[:,:corrN[1]] += realm1[:,:corrN[1]]

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

  def traintest(bimodal):
    if bimodal==FULLTRANS:
      M1 = X[:,:m2start]
      M2 = X[:,m2start:]
    elif bimodal==PARTIALTRANS:
      M1 = realm1[:,:pN[0]]
      M2 = realm2[:,:pN[1]]

    if bimodal==FULLTRANS or (bimodal==PARTIALTRANS and corrpresent):
      M1t = np.transpose(M1)
      M1tM1 = M1t @ M1
      M1tM1inv = np.linalg.inv(M1tM1)
      hatbetatrans = M1tM1inv @ M1t @ M2

    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta2 = XtXinv @ Xt @ Y

    m1test = xs[numtrain:,:m2start]
    if bimodal==FULLTRANS:
      m2test = m1test @ hatbetatrans
    elif bimodal==PARTIALTRANS and corrpresent:
      m2test = np.ones((numtest,numfeat-m2start))*X[:numtrain,m2start:].mean(axis=0)
      # print(hatbetatrans)
      # pdb.set_trace()
      m2test[:,:pN[1]] = m1test[:,:pN[0]] @ hatbetatrans
    else:
      m2test = np.ones((numtest,numfeat-m2start))*X[:numtrain,m2start:].mean(axis=0)
      m2test*= useM2avg
    testX = np.concatenate((m1test, m2test),axis=1)
    prederr = (testX @ hatbeta2) - y[numtrain:,:]
    bprederr = (np.transpose(prederr) @ prederr).squeeze()
    return bprederr

  rs = []
  if bimodal==ALL:
    for bimodal in [NAIVE,PARTIALTRANS]:
      rs.append((uprederr - traintest(bimodal))/uprederr)
  else: 
    bprederr=traintest(bimodal)
    rs.append((uprederr - bprederr)/uprederr)

  loss = uprederr/numtest
  result = [loss]
  result.extend(rs)
  return result

if __name__ == '__main__':
  data = []

  for r1 in [20,40,80]:
    for r2 in [1,4,16,64]:
      # if r2 <4: continue
      # for c2 in [20,40,60,80,100]:
      for frobNorm in [0.2,1,4,16]:
        # frobNorm = 3
        bimodal = ALL
        numits=300
        numtrain=1000
        numtest=1000
        c1 = 100
        c2 = 128

        p1,p2 = 10,1
        pN = p1,p2

        # corr2 = c2
        corrN=[r1,r2]
        # corrN=[r1,3]
        # frobNorm = 1

        show = False
        def fn(show=show):
          return linear_reg_exp(numtrain=numtrain,numtest=numtest,m1weight=[1]*c1,m2weight=[1]*c2,m2bias=[0],useM2avg=True,showBeta=False,corrN=corrN,pN=pN,frobNorm=frobNorm,bimodal=bimodal)
        # fn(True)
        name = "linear_"+bimodal+str((c1,c2))+str((r1,r2))+"_"+str(frobNorm)
        xlabels=["risk of control","risk ratio of naive","risk ratio of partial translation"]
        plotnames=[name+"control risk",name+NAIVE+" ratio",name+PARTIALTRANS+" ratio"]
        savedata=name
        show=True

        print(name)
        d = getdata(fn,numits=numits,savedata=savedata,seep=False)
        data.append(d)
        plot(d,xlabels=xlabels,plotnames=plotnames,savedata=savedata,show=False)
    
  with open("data/linearcorr_allcorr_naive.pk","wb") as f:
    pk.dump(data,f)