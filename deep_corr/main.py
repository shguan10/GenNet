import scipy as sp
import numpy as np
from models import *
import random
import time

import torch

import matplotlib.pyplot as plt
import scipy.fftpack

import pickle as pk

if __name__ == '__main__':

  numits=30
  numdata=10000
  alldata = []

  for numHlayers in [2,8,16]:
    for percent in [10,20,30,40,50]:
      
      name = str(numHlayers)+"_"+str(percent)
      print(name+"naive")

      # open the data file
      with open("mnist_split"+str(percent)+".pk","rb") as f: 
        (((trainm1,trainm2),trainlabels),((testm1,testm2),testlabels)) = pk.load(f)

      trainm1 = torch.tensor(trainm1).double()
      trainm2 = torch.tensor(trainm2).double()
      testm1 = torch.tensor(testm1).double()

      trainlabels = torch.tensor(trainlabels)
      testlabels = torch.tensor(testlabels)

      m1dim,m2dim,hiddim = trainm1.shape[1],trainm2.shape[1],100
      ydim=10

      bsize=32
      numepochs=200
      maxpatience = 20

      lr = 0.01
      momentum = 0.9

      # train the control model
      # naivemodel = Deep_Classifier(m1dim,m2dim,hiddim,numHlayers=numHlayers,ydim=ydim)
      # optimizer = torch.optim.SGD(naivemodel.parameters(),lr=lr,momentum=momentum)

      # train_dc(naivemodel,optimizer,trainm1,trainm2,trainlabels,testm1,testlabels,
      #         maxpatience = maxpatience,valratio=0.1,
      #         bsize=bsize,
      #         verbose=True,
      #         early_stop=0.001,numepochs=numepochs,
      #         datastore=alldata,translate=False,usem2train=False)
      # pdb.set_trace()

      # train the full translation model
      print("\n"+name+"full")
      fulltransmodel = Deep_Classifier(m1dim,m2dim,hiddim,numHlayers=numHlayers,ydim=ydim)
      optimizer = torch.optim.SGD(fulltransmodel.parameters(),lr=lr,momentum=momentum)

      train_dc(fulltransmodel,optimizer,trainm1,trainm2,trainlabels,testm1,testlabels,
              maxpatience = maxpatience,valratio=0.1,
              bsize=bsize,
              verbose=True,
              early_stop=0.001,numepochs=numepochs,
              datastore=alldata,translate=True,usem2train=True)      
