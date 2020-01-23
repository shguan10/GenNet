import scipy as sp
import numpy as np
from models import *

def translate_exp(numdata=1000):
    numtrain = int(0.9*numdata)
    numtest = numdata - numtrain
    numxfeat = 100
    numyfeat = 1
    numzfeat = 3
    
    xs = np.random.normal(size=(numdata,numxfeat))
    xs[:,0] = 1
    xs = xs.astype(np.float32)
    trainxs = xs[:numtrain,:]
    testxs = xs[numtrain:,:]
    trainxs = torch.tensor(trainxs)
    testxs = torch.tensor(testxs)

    realbetay = np.zeros((numxfeat,numyfeat))
    realbetay[1,:] = 1
    realbetay[2,:]=1
    y = xs @ realbetay
    y += np.random.normal(size=(y.shape))
    y = y.astype(np.float32)
    y = torch.tensor(y)
    trainy = y[:numtrain,:]
    testy = y[numtrain:,:]

    realbetaz = np.zeros((numxfeat,numzfeat))
    realbetaz[1,0] = 1
    realbetaz[2,1] = 1
    realbetaz[3,2] = 1
    z = xs @ realbetaz
    z += np.random.normal(size=(z.shape))
    z = z.astype(np.float32)
    z = torch.tensor(z)
    trainz = z[:numtrain,:]
    testz = z[numtrain:,:]

    bsize=10
    numits = int(numtrain/bsize)
    # numits = 10

    # control: one modality
    model = Model_Exp6(numxfeat,10)
    optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.001)
    print("control")
    train_model6(model,optimizer,trainxs,trainy,numits=numits,bsize=bsize)
    model.eval()
    testy = y[numtrain:,:]
    with torch.no_grad(): 
        testloss = model.forwardy(testxs,testy).numpy()
        
    testloss1 = testloss / testxs.shape[0]

    # experimental model
    model = Model_Exp6(numxfeat,10,zdim=3)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.001)
    print("pretrain")
    train_model6(model,optimizer,trainxs,trainz,predzs=True,numits=numits,bsize=bsize)
    train_model6(model,optimizer,trainxs,trainy,finetune=True,numits=numits,bsize=bsize)
    model.eval()
    testy = y[numtrain:,:]
    with torch.no_grad(): 
        testloss = model.forwardy(testxs,testy).numpy()

    testloss2 = testloss / testxs.shape[0]


    # print(testloss1)
    # print(testloss2)
    return testloss1-testloss2
