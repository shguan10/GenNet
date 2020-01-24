import scipy as sp
import numpy as np
from models import *
import random
import time

import pickle as pk

def genbetas(numM1,numM2,numH=500,numY=1,seed=None,save="sdrbetas.pk",load=None,ftype=np.float64):
    # generates the real betas for the specified parameters
    if load is not None:
        with open(load,"rb") as f: data = pk.load(f)
        return data
    if seed is not None: np.random.seed(seed)
    betaH = np.random.normal(size=(numH,numY)).astype(ftype)
    ky = np.random.normal(size=(1,numY)).astype(ftype)
    beta1 = np.random.normal(size=(numM1,numH)).astype(ftype)
    beta2 = np.random.normal(size=(numM2,numH)).astype(ftype)
    kh = np.random.normal(size=(1,numH)).astype(ftype)
    if save is not None:
        with open(save,"wb") as f: pk.dump((betaH,ky,beta1,beta2,kh),f)
    return (betaH,ky,beta1,beta2,kh)

def simple_deep_regression_exp(numdata=1000,lenM1=200,lenM2=1,lenH=500,verbose=False):
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain

    ftype = np.float64
    
    m1 = np.random.normal(size=(numdata,lenM1)).astype(ftype)
    m2 = np.random.normal(size=(numdata,lenM2)).astype(ftype)

    # print(seed)
    # time.sleep(1)

    (betaH,ky,beta1,beta2,kh) = genbetas(lenM1,lenM2,lenH,load="sdrbetas.pk",ftype=ftype)
    h = (m1@beta1 + m2@beta2 + kh)
    h = (h>0)*h
    y = h@betaH + ky
    y += np.random.normal(size=y.shape).astype(ftype)

    m1 = torch.tensor(m1)
    m2 = torch.tensor(m2)
    y = torch.tensor(y)

    trainm1 = m1[:numtrain,:]
    trainm2 = m2[:numtrain,:]
    trainy = y[:numtrain,:]

    trainymu = trainy.mean()
    trainysigma = trainy.std()
    trainy = (trainy-trainymu)/trainysigma

    testm1 = m1[numtrain:,:]
    testm2 = m2[numtrain:,:]
    testy = y[numtrain:,:]

    testy = (testy-trainymu)/trainysigma

    m2zeros_train = torch.zeros(trainm2.shape).double()
    m2zeros_test = torch.zeros(testm2.shape).double()

    # control: one modality
    if verbose: print("control")
    model = Simple_Deep_Regression(lenM1,lenM2,lenH)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                                lr=0.001)
    train_sdr(model,optimizer,trainm1,m2zeros_train,trainy,bsize=10,verbose=verbose)
    model.eval()
    with torch.no_grad(): 
        testps = model.forward(testm1,m2zeros_test)
        testloss = ((testps - testy)**2).sum().numpy()
        
    testloss1 = testloss / numtest
    if verbose: print("testloss1",testloss1)

    # experimental model
    if verbose: print("bimodal")
    model = Simple_Deep_Regression(lenM1,lenM2,lenH)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                                lr=0.001)
    train_sdr(model,optimizer,trainm1,trainm2,trainy,bsize=10,verbose=verbose)
    model.eval()
    with torch.no_grad(): 
        testps = model.forward(testm1,m2zeros_test)
        testloss = ((testps - testy)**2).sum().numpy()

    testloss2 = testloss / numtest
    if verbose:
        print("testloss2",testloss2)
        print("benefit",testloss1-testloss2)

    return testloss1-testloss2
