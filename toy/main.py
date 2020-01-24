import scipy as sp
import numpy as np

import pandas as pd

from numpy import linalg

import pdb

import matplotlib.pyplot as plt

from scipy import stats
import sklearn as sk
from sklearn import svm
from tqdm import tqdm

from translate import *
from deep_regression import *

def genpoint(stdnorm=np.random.normal,size=10):
    xs = stdnorm(size=size)
    x1 = xs
    y = ((x1[3]*(sum(x1[:3])))>0)-0.5
    x2 = np.array(xs)
    s = stdnorm(1)
    # pdb.set_trace()
    while (s*y)<=0: s = stdnorm(1)
    x2[3]=s
    return (x1,x2,y)

def genXs(numdata = 1000,numfeat = 6):
    # 0 is bias
    # 1,2,3 is X1
    # 4,5 is X2
    # 4 is from a Bernoulli, biased towards 1
    xs = np.random.normal(size=(numdata,numfeat))
    xs[:,0] = 1
    xs[:,3] /= 10
    xs[:,3] += xs[:,2]
    # xs[:,4] += xs[:,1] + xs[:,2]

    # xs[xs[:,3]>-1,3] = (xs[:,1]+xs[:,2])[xs[:,3]>-1]
    # xs[xs[:,4]>-1,3] = (xs[:,2]+xs[:,3])[xs[:,4]>-1]

    return (xs)

def gendata(numdata=1000):
    return [genpoint() for _ in range(numdata)]

def getLOS():
    df = pd.read_csv("LOS.txt",sep="\t")
    data = df.to_numpy()
    (numdata,numfeat) = data.shape
    y = data[:,1].reshape((numdata,1))
    X = np.empty((numdata,numfeat-1))
    X[:,1:] = data[:,2:]
    X[:,0] = np.ones((numdata,))
    return (X,y)

def getEstimatorError(std=1,lmb=10):
    numdata = 1000
    numfeat = 10

    X =  genXs(numdata=numdata,numfeat=numfeat)
    testX = X[-100:,:]
    X = X[:-100,:]
    realbeta = np.zeros((numfeat,1))
    realbeta[1:4,:] = 1
    y = (X @ realbeta).reshape(-1,1)
    # y = (X[:,1] + X[:,2]).reshape(numdata,1)
    y += np.random.normal(0,std,size=y.shape)
    
    # X,y = getLOS()
    # numdata,numfeat = X.shape
    
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    yt = np.transpose(y)

    # first do simple linear regression
    hatbeta = XtXinv @ Xt @ y
    H = X @ XtXinv @ Xt

    # print("OLS SSE: ",SSE)
    # print("OLS hatbeta: ",hatbeta)
    # print("OLS hatbeta l2: ", np.transpose(hatbeta) @ hatbeta)
    errorbeta = hatbeta - realbeta
    olsdiff = (np.transpose(errorbeta)@ errorbeta).squeeze()
    # print("l2 diff from realbeta: ",)
    prederr = (testX @ errorbeta)
    olsSSEp = (np.transpose(prederr) @ prederr).squeeze()

    # now do ridge regression
    XtXpEinv = np.linalg.inv(XtX + lmb*np.eye(numfeat))
    hatbeta = XtXpEinv @ Xt @ y
    H = X @ XtXpEinv @ Xt

    # print("ridge SSE: ",SSE)
    # print("ridge hatbeta: ",hatbeta)
    # print("ridge hatbeta l2: ", np.transpose(hatbeta) @ hatbeta)
    errorbeta = hatbeta - realbeta
    ridgediff = (np.transpose(errorbeta)@ errorbeta).squeeze()

    prederr = (testX @ errorbeta)
    ridgeSSEp = (np.transpose(prederr) @ prederr).squeeze()

    return olsdiff,ridgediff,olsSSEp,ridgeSSEp

def exp1():
    # this will be the naive experiment, testing if the translation
    # amounts to a l2 regularization
    numdata = 1000
    numfeat = 3

    xs = genXs().astype(np.float32)

    trainxs = xs[:-100,:]
    testxs = xs[-100:,:]

    realbeta = np.zeros((numfeat*2,1)).astype(np.float32)
    realbeta[1:4,:] = 1 # x1 and x2 have complementary info

    # OLS
    model = OLS(6)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.001)
    trainxs = torch.tensor(trainxs)
    testxs = torch.tensor(testxs)
    realbeta = torch.tensor(realbeta)

    # pdb.set_trace()
    olsSSEp = traintest_exp1(model,optimizer,trainxs,testxs,realbeta)

    # experimental model
    model = Translate(3,3,3)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.001,
                                # momentum=0.9,
                                weight_decay=0.2)
    optimizer.zero_grad()
    
    tran_testloss = traintest_exp1(model,optimizer,trainxs,testxs,realbeta)

    return olsSSEp,tran_testloss

def traintest_exp1(model,optimizer,trainxs,testxs,realbeta,numdata=1000):
    std=1
    # train for 10 epochs
    for _ in range(1000):
        bsize=100
        for start in range(int(numdata/bsize)):
            data = trainxs[start:start+bsize,:]
            x1 = data[:,:3]
            x2 = data[:,3:]
            y = (data @ realbeta).reshape(-1,1)
            y += torch.tensor(np.random.normal(0,std,size=y.shape).astype(np.float32))

            loss = model.forward(x1,x2,y) / bsize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    y = (testxs @ realbeta).reshape(-1,1)
    y += torch.tensor(np.random.normal(0,std,size=y.shape).astype(np.float32))
    x1 = testxs[:,:3]
    x2 = testxs[:,3:]
    model.eval()
    with torch.no_grad(): 
        testloss = model.forward1(x1,x2,y).numpy()
        
    return testloss / testxs.shape[0]

def exp3(numdata=1000,corrp = 0.01,cutoffp = 0.8,x4weight=1):
    # this will be the unimodal task experiment with a weak form of correlation between X4 and X2
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain
    numfeat = 5
    
    cutoff = sp.stats.norm(0,1).ppf(cutoffp)
    select = np.random.binomial(1,corrp,size=(numdata,1))

    xs = np.random.normal(size=(numdata,numfeat))
    xs[:,0] = 1
    xs[:,4] = ((xs[:,4]<xs[:,2])[:,None]*select + (xs[:,4]<cutoff)[:,None]*(1-select)).squeeze()
    # xs[:,4] += ((xs[:,4]<xs[:,2])[:,None]*select).squeeze()
    realbeta = np.zeros((numfeat,1))
    realbeta[1:3,:] = 10
    realbeta[4,:] = x4weight
    y = xs @ realbeta
    y += np.random.normal(size=y.shape)

    # do the first model
    X = xs[:numtrain,:4]
    Y = y[:numtrain,:]
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta = XtXinv @ Xt @ Y

    testX = xs[numtrain:,:4]
    prederr = (testX @ hatbeta) - y[numtrain:,:]
    uprederr = (np.transpose(prederr) @ prederr).squeeze()
    # print("control error: ",uprederr)

    # do the second model
    X = xs[:numtrain,:]
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta = XtXinv @ Xt @ Y

    testX = np.concatenate((xs[numtrain:,:4], np.ones((numtest,1))*X[:,4].mean()),axis=1)
    prederr = (testX @ hatbeta) - y[numtrain:,:]
    bprederr = (np.transpose(prederr) @ prederr).squeeze()
    # print("bimodal error: ",bprederr)
    # print("bimodal benefit: ",uprederr - bprederr)
    return uprederr - bprederr

def exp4(numdata=1000,m2weight=(1,1,0,0)):
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain
    numfeat = 8
    
    xs = np.random.normal(size=(numdata,numfeat))
    xs[:,0] = 1
    # x5 will be correlated in some way with x2
    xs[:,5] += xs[:,2]
    realbeta = np.zeros((numfeat,1))
    realbeta[1:3,:] = 10
    realbeta[4:,:] = np.array(m2weight)[:,None]
    y = xs @ realbeta
    y += np.random.normal(size=y.shape)

    # do the first model
    X = xs[:numtrain,:4]
    Y = y[:numtrain,:]
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta = XtXinv @ Xt @ Y

    testX = xs[numtrain:,:4]
    prederr = (testX @ hatbeta) - y[numtrain:,:]
    uprederr = (np.transpose(prederr) @ prederr).squeeze()

    # do the second model
    X = xs[:numtrain,:]
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta = XtXinv @ Xt @ Y

    testX = np.concatenate((xs[numtrain:,:4], 
                            np.ones((numtest,numfeat-4))*X[:,4:].mean(axis=0)),axis=1)
    prederr = (testX @ hatbeta) - y[numtrain:,:]
    bprederr = (np.transpose(prederr) @ prederr).squeeze()
    return uprederr - bprederr

def exp5(numdata=1000,m2weight=(1,0.01,1,0)):
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain
    numfeat = 8
    
    xs = np.random.normal(size=(numdata,numfeat))
    xs[:,0] = 1
    # x5 will be correlated in some way with x2
    # xs[:,5] /= 10
    # xs[:,5] += xs[:,2]/10
    realbeta = np.zeros((numfeat,1))
    realbeta[1:3,:] = 10
    realbeta[4:,:] = np.array(m2weight)[:,None]
    y = xs @ realbeta
    # y += np.random.normal(size=(y.shape))/16
    y = (y>0).astype(np.int64).squeeze()

    # do the first model
    X = xs[:numtrain,:4]
    Y = y[:numtrain]
    clf = sk.svm.SVC(kernel='linear')
    clf.fit(X,Y)

    testX = xs[numtrain:,:4]
    prederr = (clf.predict(testX)) - y[numtrain:]
    uprederr = (np.transpose(prederr) @ prederr).squeeze()

    # do the second model
    X = xs[:numtrain,:]
    clf = sk.svm.SVC(kernel='linear')
    clf.fit(X,Y)

    testX = np.concatenate((xs[numtrain:,:4], 
                            np.ones((numtest,numfeat-4))*X[:,4:].mean(axis=0)),axis=1)
    prederr = (clf.predict(testX)) - y[numtrain:]
    bprederr = (np.transpose(prederr) @ prederr).squeeze()
    benefit = uprederr - bprederr
    return benefit

def exp7(numdata=1000,corrp = 0.01,x4weight=1,x1x4weight=1):
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain
    numfeat = 5
    
    xs = np.random.normal(size=(numdata,numfeat))
    xs[:,0] = 1
    
    realbeta = np.zeros((numfeat+1,1))
    realbeta[1:3,:] = 10
    realbeta[4,:] = x4weight
    realbeta[5,:] = x1x4weight

    xs = np.concatenate((xs,(xs[:,4]*xs[:,1]).reshape(numdata,-1)),axis=1)

    y = xs @ realbeta
    y += np.random.normal(size=(y.shape))

    # do the first model
    X = xs[:numtrain,:4]
    Y = y[:numtrain,:]
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta = XtXinv @ Xt @ Y

    testX = xs[numtrain:,:4]
    prederr = (testX @ hatbeta) - y[numtrain:,:]
    uprederr = (np.transpose(prederr) @ prederr).squeeze()

    # do the second model
    X = xs[:numtrain,:]
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta = XtXinv @ Xt @ Y

    X4avg = np.ones((numtest,numfeat-4))*X[:numtrain,4].mean(axis=0)
    testX = np.concatenate((xs[numtrain:,:4], 
                            X4avg,
                            X4avg*(xs[numtrain:,1]).reshape(-1,1)),axis=1)
    prederr = (testX @ hatbeta) - y[numtrain:,:]
    bprederr = (np.transpose(prederr) @ prederr).squeeze()
    return uprederr - bprederr

BENEFIT = 0
DIFFNORM = 1
BIAS = 2
NORMDIFF = 3
DIFFDOTREAL = 4

def linear_reg_exp(numdata=1000,m1weight=(1,),m2weight=(1,),corrN=0,corrScale=1,metric=BENEFIT,m2bias=[0],useM2avg=True,showBeta=False):
    assert corrN < len(m2weight)
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain
    C1 = len(m1weight)
    C2 = len(m2weight)
    numfeat = 1+C1+C2
    m2start = 1+C1
    m2corrstart = m2start + C2 - corrN

    xs = np.random.normal(size=(numdata,numfeat)).astype(np.float64)
    xs[:,0] = 1
    xs[:,m2start:]+=np.array(m2bias).reshape(1,-1)
    xs[:,m2corrstart:] += xs[:,:corrN] * corrScale
    
    m2bias = np.array(m2bias).mean()
    # realbeta = [-m2bias]+list(m1weight) + list(m2weight)
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
    Xt = np.transpose(X)
    XtX = Xt @ X
    XtXinv = np.linalg.inv(XtX)
    hatbeta2 = XtXinv @ Xt @ Y

    if showBeta:
        plt.plot(hatbeta,'g',label="control hatbeta")
        plt.plot(hatbeta2,'r',label="mm hatbeta")
        plt.plot(realbeta,"b--",label="real beta")
        plt.xlabel("index of (hat)beta")
        plt.legend()
        plt.title("hatbeta for M2 ~ N(%.1f,1)"%m2bias)
        plt.show()
        pdb.set_trace()

    m2avg = np.ones((numtest,numfeat-m2start))*X[:numtrain,m2start:].mean(axis=0)
    m2avg*= useM2avg
    testX = np.concatenate((xs[numtrain:,:m2start],
                            m2avg),axis=1)
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

def svm_exp(numdata=1000,m1weight=[1],m2weight=[1],m2bias=[1]):
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain
    numfeat = 1+len(m1weight)+len(m2weight)
    m2start = 1+len(m1weight)

    xs = np.random.normal(size=(numdata,numfeat)).astype(np.float64)
    xs[:,0] = 1
    xs[:,m2start:]+=np.array(m2bias).reshape(1,-1)
    
    realbeta = [1]+list(m1weight) + list(m2weight)
    realbeta = np.array(realbeta).reshape(-1,1).astype(np.float64)

    y = xs @ realbeta
    y += np.random.normal(size=(y.shape))
    y -= np.array(m2bias).mean()
    y = (y>0).astype(np.int64).squeeze()

    # do the first model
    X = xs[:numtrain,:m2start]
    Y = y[:numtrain]
    clf = sk.svm.SVC(kernel='linear')
    clf.fit(X,Y)

    testX = xs[numtrain:,:m2start]
    prederr = (clf.predict(testX)) - y[numtrain:]
    uprederr = (np.transpose(prederr) @ prederr).squeeze()

    # do the second model
    X = xs[:numtrain,:]
    clf = sk.svm.SVC(kernel='linear')
    clf.fit(X,Y)

    m2avg = np.ones((numtest,numfeat-m2start))*X[:numtrain,m2start:].mean(axis=0)
    testX = np.concatenate((xs[numtrain:,:m2start],
                            m2avg),axis=1)
    prederr = (clf.predict(testX)) - y[numtrain:]
    bprederr = (np.transpose(prederr) @ prederr).squeeze()
    return uprederr - bprederr

def main(fn,numits=1000,xlabel="sample value"):
    N = numits
    data = np.array([])
    for it in range(N):
        res = fn()
        data = np.concatenate((data,[res]))
        mu = data.mean()
        sigma = data.std(ddof=1) if it>0 else 0
        sigma /= np.sqrt(it+1)
        t = sp.stats.t(it)
        p = t.cdf(-mu/sigma) if it>0 else 1
        print("it: ",it,"/",N-1,
              ", mean: %.4f" %mu,
              ", std: %.4f"%sigma,
              ", p value: %.4f"%p,end="\r",flush=True)
    print("\n")
    lo = mu+(t.ppf(0.05)*sigma)
    # lo = mu+(t.ppf(0.025)*sigma)
    # hi = mu+(t.ppf(0.975)*sigma)
    xs = np.arange(N)
    ones = np.ones(N)
    _=plt.hist(data,bins=int(N/10))
    plt.ylabel("num of samples")
    plt.xlabel("sample value of "+xlabel)
    plt.axvline(x=lo,color="r",label="95% confidence lower bound for sample mean")
    plt.legend()
    plt.title("histogram of "+xlabel)
    # plt.plot(xs,data,'rs',xs,ones*lo,'k-',xs,np.zeros(xs.shape),'w-')
    # plt.plot(xs,data,'rs',xs,ones*lo,'g--',xs,ones*hi,'g--')
    plt.show()

if __name__ == '__main__':
    # olse,tte = exp1()
    # print(olse)
    # print(tte)
    # print("ols - tt, ",olse-tte)
    metric = BENEFIT
    if metric==BENEFIT: xlabel="mm benefit"
    elif metric==DIFFNORM:  xlabel="diff in norm (control - mm)"
    elif metric==NORMDIFF:  xlabel="norm of the diff in hatbeta of M1"
    elif metric==BIAS: xlabel="diff in hatbeta[0] (control - mm)"
    elif metric==DIFFDOTREAL: xlabel="diff in hatbeta dot product with realbeta (mm - control)"

    numits=30

    corrN=199
    c2 = 1+corrN
    corrScale = 0.0625

    lenM1=10
    lenM2=10
    lenH = 100
    actualH = 10
    numHlayers = 2

    def fn(show=False):
        # return exp3(numdata=10000,corrp = 0.1,cutoffp = 0.5,x4weight=1)
        # return exp4(numdata=10000,m2weight=(1,0.01,1,0))
        # return exp5(numdata=10000,m2weight=(1,0.01,1,0))
        # return exp7(x4weight=1,x1x4weight=0)
        # return linear_reg_exp(numdata=10000,m1weight=[1]*100,m2weight=[1]*c2,corrN=corrN,corrScale=corrScale,m2bias=[0],useM2avg=True,showBeta=False,metric=metric)
        # return svm_exp(numdata=1000,m1weight=[1]*200,m2weight=[30],m2bias=[1])
        # return translate_exp(numdata=1000)
        # return simple_deep_regression_exp(lenM1=lenM1,lenM2=lenM2,lenH=lenH,verbose=True)
        return deep_regression_exp(lenM1=lenM1,lenM2=lenM2,lenH=lenH,numHlayers=numHlayers,verbose=True)
    genbetas_deep(lenM1,lenM2,actualH,numHlayers=numHlayers,save="sdrbetas.pk")
    # fn(True)
    main(fn,numits=numits,xlabel=xlabel)
