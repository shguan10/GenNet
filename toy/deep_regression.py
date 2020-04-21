import scipy as sp
import numpy as np
from models import *
import random
import time

import matplotlib.pyplot as plt
import scipy.fftpack

import pickle as pk
from measure import plot

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

def deep_regression_exp(numdata=1000,diffvar=False,maxpatience = 20,lenM1=200,lenM2=1,lenH=500,numHlayers=1,verbose=False,bsize=32,numepochs=200,show=False,translate=False,corrN=[0,0],frobNorm=1):
  numtrain = int(0.9*numdata)
  numtest = numdata - numtrain
  numval = numtest

  ftype = np.float64
  
  m1 = np.random.normal(size=(numdata,lenM1)).astype(ftype)
  m2 = np.random.normal(size=(numdata,lenM2)).astype(ftype)

  corrpresent = corrN[1] > 0

  if corrpresent:
    transmatrix = np.random.normal(size=corrN)
    transmatrix *= frobNorm / np.linalg.norm(transmatrix)

    # now generate m2 from m1
    addm2 = m1[:,:corrN[0]] @ transmatrix

    m2[:,:corrN[1]] += addm2

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
  m2zeros_test = torch.ones((numtest,lenM2)).double().cuda()*trainm2.mean(axis=0)

  # control: one modality
  if verbose: print("\ncontrol")
  model = Deep_Regression(lenM1,lenM2,lenH,numHlayers=numHlayers).cuda()
  optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True,
                lr=0.001)
  traindata = [] if (show or diffvar) else None
  train_dr(model,optimizer,trainm1,m2zeros_train,trainy,
        maxpatience=maxpatience,
        numval=numval,testset=(testm1,m2zeros_test,testy),
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
  optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.001)
  traindata2 = [] if (show or diffvar) else None
  if translate:
    with torch.no_grad():
      hatbetatrans = \
        (trainm1.t() @ trainm1).inverse() @ trainm1.t() @ trainm2
  train_dr(model,optimizer,trainm1,trainm2,trainy,
        maxpatience=maxpatience,
        numval=numval,testset=(testm1,m2zeros_test,testy),
        bsize=bsize,verbose=verbose,numepochs=numepochs,datastore=traindata2,translate=hatbetatrans)

  model.eval()
  with torch.no_grad(): 
    if translate: m2zeros_test = testm1 @ hatbetatrans
    testps = model.forward(testm1,m2zeros_test)
    testloss = ((testps - testy)**2).sum().cpu().numpy()

  testloss2 = testloss / numtest
  
  traindata = np.array(traindata)
  traindata2 = np.array(traindata2)
  
  if diffvar: difftestvar = traindata[:,1].var()-traindata2[:,1].var()

  if verbose:
    print("testloss2",testloss2)
    print("benefit",testloss1-testloss2)
    if diffvar: print("difftestvar",difftestvar)

  if show:
    # plot the training curves

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('training loss per sample')
    controltrain, = ax1.plot(traindata[:,0], color = 'red')
    bitrain, = ax1.plot(traindata2[:,0], color='blue')
    ax1.tick_params(axis='y')

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # ax2.set_ylabel('testing loss per sample')  # we already handled the x-label with ax1
    controltest, = ax1.plot(traindata[:,1], color = 'orange')
    bitest, = ax1.plot(traindata2[:,1], color='cyan')
    # ax2.tick_params(axis='y')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend((controltrain,bitrain,controltest,bitest),('control val','bimodal val','control testing','bimodal testing'))
    # plt.savefig("figs/"+str(random.random())[2:]+".jpg")
    plt.show()
    plt.clf()

    # plot the fourier transform
    plt.ylabel('amplitude')
    plt.xlabel('frequency (1/epoch)')
    
    N = len(traindata[:,1])
    yfuni = scipy.fftpack.fft(traindata[:,1])
    xf = np.linspace(0,1/2,N/2)
    yfuni = 2/N*np.abs(yfuni[:len(xf)])
    plt.plot(xf,yfuni,color='orange',label="control testing")

    N = len(traindata2[:,1])
    yfbi = scipy.fftpack.fft(traindata2[:,1])
    xf = np.linspace(0,1/2,N/2)
    yfbi = 2/N*np.abs(yfbi[:len(xf)])
    plt.plot(xf,yfbi,color="cyan",label="bimodal testing")
    plt.legend()
    plt.show()
    pdb.set_trace()


  return [testloss1-testloss2,testloss1] if not diffvar else [testloss1-testloss2,testloss1,difftestvar] 


def getdata(fn,numits=1000,savedata=None):
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
        print("\nsaved data to ",savedata)
    
    print("\n")
    return npdata

def plot(npdata,xlabels=["sample value"],plotnames=None,savedata=None,show=True):
  N = len(npdata)
  mu = npdata.mean(axis=0)
  sigma = npdata.std(ddof=1,axis=0)
  sigma /= np.sqrt(npdata.shape[0])
  
  t = sp.stats.t(N)
  lo = mu+(t.ppf(0.05)*sigma)
  for ind,lab in enumerate(xlabels):
      xs = np.arange(N)
      ones = np.ones(N)
      _=plt.hist(npdata[:,ind],bins=max(int(N/10),1))
      plt.ylabel("num of samples")
      plt.xlabel("sample value of "+lab)
      plt.axvline(x=lo[ind],color="r",label="95%% confidence lower bound for sample mean (%f)"%lo[ind])
      plt.legend()
      plt.title("histogram of "+lab)
      if show: plt.show()
      else: plt.savefig("figs/"+plotnames[ind]+".jpg")
      plt.clf()

if __name__ == '__main__':
  xlabels = ["mm benefit","testloss1"]

  numits=20
  numdata=10000
  translate=True

  for frobNorm in np.linspace(8,12,num=10,endpoint=False):
    # if frobNorm<=8: continue
    print("#######\nfrobNorm "+str(frobNorm))
    lenM1=100
    lenM2=10
    lenH = 100
    actualH = 10
    numHlayers = 1

    corrN=[lenM1,lenM2]

    name="frobInc_"+str(lenM1)+"_"+str(lenM2)+"_"+str(lenH)+"_"+str(actualH)+"_"+str(numHlayers)+"_"+str(frobNorm)+"_"+str(translate)

    print(name)

    bsize=32
    numepochs=400
    maxpatience = 20

    show = False
    def fn(show=show):
      genbetas_deep(lenM1,lenM2,actualH,numHlayers=numHlayers,save="drbetas.pk")
      return deep_regression_exp(numdata=numdata,diffvar=False,maxpatience=maxpatience,lenM1=lenM1,lenM2=lenM2,lenH=lenH,numHlayers=numHlayers,verbose=True,bsize=bsize,numepochs=numepochs,show=show,translate=translate,corrN=corrN,frobNorm=frobNorm)
    # fn(True)
    plotnames=[name+"benefit",name+"loss"]
    savedata = name
    data = getdata(fn,numits=numits,savedata=savedata)
    plot(data,xlabels=xlabels,plotnames=plotnames,savedata=savedata,show=False)