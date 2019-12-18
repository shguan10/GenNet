import scipy as sp
import numpy as np

import pandas as pd

from numpy import linalg

import pdb

import matplotlib.pyplot as plt

from models import *


"""
The scenario we are testing is the simple linear regression vs the
translation via intermediary vector to final classification task.

"""

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
    olsSSEp = traintest(model,optimizer,trainxs,testxs,realbeta)

    # experimental model
    model = Translate(3,3,3)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.001,
                                # momentum=0.9,
                                weight_decay=0.2)
    optimizer.zero_grad()
    
    tran_testloss = traintest(model,optimizer,trainxs,testxs,realbeta)

    return olsSSEp,tran_testloss

def traintest(model,optimizer,trainxs,testxs,realbeta,numdata=1000):
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

def main():
    for std in [1]:
        numpoints = 100
        avg = np.ones(numpoints)
        data = [getEstimatorError(1,2.5) for s in range(numpoints)]
        data = np.array(data)
        olserrs = data[:,0]
        olsavg = avg * olserrs.mean()
        ridgeerrs = data[:,1]
        ridgeavg = avg * ridgeerrs.mean()
        op = data[:,2].mean()
        rp = data[:,3].mean()
        print("obias - rbias",(olsavg - ridgeavg)[0])(model.w1h.weight,model.w2h.weight)
        print("ostd - rstd",olserrs.std() - ridgeerrs.std())
        print("op - rp",(op - rp))
        # np.save("olserrs.npy",olserrs)
        # np.save("ridgeerrs.npy",ridgeerrs)
        xs = np.arange(numpoints)
        plt.plot(xs,olserrs,'rs',xs,ridgeerrs,'bs',
                 xs,olsavg,"r--",xs,ridgeavg,"b--")
        plt.show()

if __name__ == '__main__':
    olse,tte = exp1()
    print(olse)
    print(tte)
    print("ols - tt, ",olse-tte)
