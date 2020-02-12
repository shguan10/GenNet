import scipy as sp
import numpy as np
from models import *
import random
import time

import matplotlib.pyplot as plt

import pickle as pk

def genbetas_simple(numM1,numM2,numH=500,numY=1,seed=None,save="sdrbetas.pk",load=None,ftype=np.float64):
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

    (betaH,ky,beta1,beta2,kh) = genbetas_simple(lenM1,lenM2,lenH,load="sdrbetas.pk",ftype=ftype)
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

    testm1 = m1[numtrain:,:]
    testm2 = m2[numtrain:,:]
    testy = y[numtrain:,:]

    trainymu = trainy.mean()
    trainysigma = trainy.std()
    trainy = (trainy-trainymu)/trainysigma
    testy = (testy-trainymu)/trainysigma

    m2zeros_train = torch.zeros(trainm2.shape).double()
    m2zeros_test = torch.zeros(testm2.shape).double()

    # control: one modality
    if verbose: print("\ncontrol")
    model = Deep_Regression(lenM1,lenM2,lenH)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                                lr=0.001)
    train_dr(model,optimizer,trainm1,m2zeros_train,trainy,bsize=10,verbose=verbose)
    model.eval()
    with torch.no_grad(): 
        testps = model.forward(testm1,m2zeros_test)
        testloss = ((testps - testy)**2).sum().numpy()
        
    testloss1 = testloss / numtest
    if verbose: print("testloss1",testloss1)

    # experimental model
    if verbose: print("bimodal")
    model = Deep_Regression(lenM1,lenM2,lenH)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                                lr=0.001)
    train_dr(model,optimizer,trainm1,trainm2,trainy,bsize=10,verbose=verbose)
    model.eval()
    with torch.no_grad(): 
        testps = model.forward(testm1,m2zeros_test)
        testloss = ((testps - testy)**2).sum().numpy()

    testloss2 = testloss / numtest
    if verbose:
        print("testloss2",testloss2)
        print("benefit",testloss1-testloss2)

    return testloss1-testloss2

def genbetas_deep(lenM1,lenM2,lenH,numHlayers=1,lenY=1,seed=None,save="sdrbetas.pk",load=None,ftype=np.float64):
    # generates the real betas for the specified parameters
    if load is not None:
        with open(load,"rb") as f: data = pk.load(f)
        return data
    if seed is not None: np.random.seed(seed)
    paramsH = []
    for layer in range(numHlayers):
        lendest = lenY if layer==numHlayers-1 else lenH
        beta = np.random.normal(size=(lenH,lendest)).astype(ftype)
        k = np.random.normal(size=(1,lendest)).astype(ftype)
        paramsH.append((beta,k))
    beta1 = np.random.normal(size=(lenM1,lenH)).astype(ftype)
    beta2 = np.random.normal(size=(lenM2,lenH)).astype(ftype)
    kh = np.random.normal(size=(1,lenH)).astype(ftype)

    params = (paramsH,beta1,beta2,kh)
    if save is not None:
        with open(save,"wb") as f: pk.dump(params,f)
    return params

def deep_regression_exp(numdata=1000,lenM1=200,lenM2=1,lenH=500,numHlayers=1,verbose=False,bsize=32,numepochs=200,show=False):
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain

    ftype = np.float64
    
    m1 = np.random.normal(size=(numdata,lenM1)).astype(ftype)
    m2 = np.random.normal(size=(numdata,lenM2)).astype(ftype)

    # print(seed)
    # time.sleep(1)

    (paramHs,beta1,beta2,kh) = genbetas_deep(lenM1,lenM2,lenH,numHlayers=numHlayers,load="drbetas.pk",ftype=ftype)
    h = (m1@beta1 + m2@beta2 + kh)
    h = (h>0)*h
    for ind,(beta,k) in enumerate(paramHs):
        h = h@beta + k
        if ind+1<len(paramHs): h = (h>0)*h
    y = h+np.random.normal(size=h.shape).astype(ftype)

    m1 = torch.tensor(m1).cuda()
    m2 = torch.tensor(m2).cuda()
    y = torch.tensor(y).cuda()

    trainm1 = m1[:numtrain,:]
    trainm2 = m2[:numtrain,:]
    trainy = y[:numtrain,:]

    testm1 = m1[numtrain:,:]
    testm2 = m2[numtrain:,:]
    testy = y[numtrain:,:]

    trainymu = trainy.mean()
    trainysigma = trainy.std()
    trainy = (trainy-trainymu)/trainysigma
    testy = (testy-trainymu)/trainysigma

    m2zeros_train = torch.zeros(trainm2.shape).double().cuda()
    m2zeros_test = torch.zeros(testm2.shape).double().cuda()

    # control: one modality
    if verbose: print("\ncontrol")
    model = Deep_Regression(lenM1,lenM2,lenH,numHlayers=numHlayers).cuda()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                                lr=0.001)
    traindata = [] if show else None
    train_dr(model,optimizer,trainm1,m2zeros_train,trainy,testset=(testm1,m2zeros_test,testy),
                bsize=bsize,verbose=verbose,numepochs=numepochs,datastore=traindata)
    model.eval()
    with torch.no_grad(): 
        testps = model.forward(testm1,m2zeros_test)
        testloss = ((testps - testy)**2).sum().cpu().numpy()
        
    testloss1 = testloss / numtest
    if verbose: print("testloss1",testloss1)

    # experimental model
    if verbose: print("bimodal")
    model = Deep_Regression(lenM1,lenM2,lenH,numHlayers=numHlayers).cuda()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                                lr=0.001)
    traindata2 = [] if show else None
    train_dr(model,optimizer,trainm1,trainm2,trainy,testset=(testm1,m2zeros_test,testy),
                bsize=bsize,verbose=verbose,numepochs=numepochs,datastore=traindata2)
    model.eval()
    with torch.no_grad(): 
        testps = model.forward(testm1,m2zeros_test)
        testloss = ((testps - testy)**2).sum().cpu().numpy()

    testloss2 = testloss / numtest
    if verbose:
        print("testloss2",testloss2)
        print("benefit",testloss1-testloss2)

    if show:
        # plot the training curves
        traindata = np.array(traindata)
        traindata2 = np.array(traindata2)

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training loss per sample')
        controltrain, = ax1.plot(traindata[:,0], color = 'red')
        bitrain, = ax1.plot(traindata2[:,0], color='blue')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('testing loss per sample')  # we already handled the x-label with ax1
        controltest, = ax2.plot(traindata[:,1], color = 'orange')
        bitest, = ax2.plot(traindata2[:,1], color='cyan')
        ax2.tick_params(axis='y')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend((controltrain,bitrain,controltest,bitest),('control training','bimodal training','control testing','bimodal testing'))
        plt.savefig("figs/"+str(random.random())[2:]+".jpg")
        # plt.show()


    return [testloss1-testloss2,testloss1]
