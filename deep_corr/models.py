import torch
import pdb
import numpy as np

# torch.manual_seed(0)
# np.random.seed(8944)

class Deep_Classifier(torch.nn.Module):
  def __init__(self,m1dim,m2dim,hiddim,numHlayers=1,ydim=1):
    torch.nn.Module.__init__(self)
    self.beta1_kh = torch.nn.Linear(m1dim,hiddim,bias=True).double()
    self.beta2 = torch.nn.Linear(m2dim,hiddim,bias=False).double()
    paramsH = []
    for layer in range(numHlayers):
      lendest = ydim if layer==numHlayers-1 else hiddim
      l = torch.nn.Linear(hiddim,lendest,bias=True).double()
      paramsH.append(l)
    self.paramsH = torch.nn.ModuleList(paramsH)
    
  def forward(self,m1,m2):
    h = self.beta1_kh(m1) + self.beta2(m2)
    h = torch.relu(h)
    for ind,m in enumerate(self.paramsH):
      h = m(h)
      if ind<len(self.paramsH)-1: h = torch.relu(h)
    return h

def check_num_correct(logits,labels):
  # logits has shape (bsize,numclasses)
  return np.array(np.argmax(logits,axis=1)==labels).sum()

def train_dc(model,optimizer,trainm1,trainm2,trainlabels,testm1,testlabels,maxpatience = 20,valratio=0.1,bsize=10,verbose=False,early_stop=0.001,numepochs=200,datastore=None,translate=False,usem2train=False):
  """
  translate is whether to translate during testing
  
  usem2train denotes whether you want to use m2 during training
  labels must be not be one-hot encoded
  """
  numtrain = trainm1.shape[0]
  numval = int(valratio*numtrain)
  numtrain -= numval
  numbatches = int(numtrain / bsize)

  # pdb.set_trace()

  valzeros = torch.ones((numval,trainm2.shape[1])).double().cpu()*trainm2.mean(axis=0)
  testzeros = torch.ones((testm1.shape[0],trainm2.shape[1])).double().cpu()*trainm2.mean(axis=0)
  if translate:
    ntrainm1 = trainm1.cpu().numpy()
    ntrainm2 = trainm2.cpu().numpy()
    ntestm1 = testm1.cpu().numpy()
    translate = np.linalg.pinv(np.transpose(ntrainm1) @ ntrainm1)@np.transpose(ntrainm1)@ntrainm2

    valzeros[:,:translate.shape[1]] = torch.tensor(ntrainm1[numtrain:,:] @ translate)
    testzeros[:,:translate.shape[1]] = torch.tensor(ntestm1 @ translate)

  if not usem2train:
    trainm2 = torch.ones(trainm2.shape).double().cpu()*trainm2.mean(axis=0)

  patience = maxpatience
  prevmin = None
  criterion = torch.nn.CrossEntropyLoss()
  for epoch in range(numepochs):
    epochloss = 0
    trainacc = 0
    idxs = (np.random.rand(numbatches,bsize)*numtrain).astype(np.int64)

    for batch in idxs:
      optimizer.zero_grad()
      m1 = trainm1[batch]
      m2 = trainm2[batch]
      by = trainlabels[batch]
      # by = by.reshape(-1,labels.shape[1])

      py = model.forward(m1,m2)
      loss = criterion.forward(py,by)
      epochloss+=loss

      loss.backward()
      optimizer.step()

      trainacc += check_num_correct(py.detach().numpy(),by.numpy())

    trainacc /= numtrain
    avgsampleloss = epochloss/numtrain

    with torch.no_grad():
      valps = model.forward(trainm1[numtrain:],valzeros)
      valloss = criterion(valps,trainlabels[numtrain:])/numval
      valacc = check_num_correct(valps,trainlabels[numtrain:])/numval
    if datastore is not None: 
      with torch.no_grad():
        numtest = len(testlabels)
        testps = model.forward(testm1,testzeros)
        testloss = criterion(testps,testlabels)/numtest
        testacc = check_num_correct(testps,testlabels)/numtest

      datastore.append(((avgsampleloss,valloss,testloss),(trainacc,valacc,testacc)))

    if verbose:
      print("epoch: ",epoch,"/",numepochs-1,
        ", avgsampleloss: %.4f" %avgsampleloss, 
        ", trainacc: %.4f" %trainacc, 
        end="\r")
    
    if prevmin is None or valloss < prevmin: 
      patience = maxpatience
      prevmin = valloss
    else: patience -= 1
    if patience <=0: break
  print("\ntestloss: %.4f" %testloss,
        "testacc: %.4f" %testacc)
  # if verbose: print("\n")