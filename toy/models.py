import torch
import pdb
import numpy as np

# torch.manual_seed(0)
# np.random.seed(8944)

class Translate(torch.nn.Module):
    def __init__(self,numx1s,numx2s,numhs):
        torch.nn.Module.__init__(self)
        self.w1h = torch.nn.Linear(numx1s,numhs)
        self.w2h = torch.nn.Linear(numx2s,numhs)
        self.wh1 = torch.nn.Linear(numhs,numx1s)
        self.wh2 = torch.nn.Linear(numhs,numx2s)
        self.who = torch.nn.Linear(numhs,1)

    def forward1(self,x1,x2,y):
        H1 = self.w1h(x1)
        H2 = self.w2h(x2)
        L4 = self.who(H1+H2)*0.5 - y

        # h = self.w1h(x1)
        # return self.who(h)
        loss = (L4.t() @ L4).squeeze()
        return loss

    def forward(self,X1,X2,Y):
        # assumes this is a batch of size greater than 1
        H1 = self.w1h(X1)
        H2 = self.w2h(X2)
        X2p = self.wh2(H1)
        X1p = self.wh1(H2)
        H1p = self.w1h(X1p)
        H2p = self.w2h(X2p)
        X1pp = self.wh1(H2p)
        X2pp = self.wh2(H1p)
        L1 = X2p - X2
        L2 = X1p - X1
        L3 = H1 - H2
        L4 = self.who(H1+H2)*0.5 - Y
        L5 = X1-X1pp
        L6 = X2-X2pp

        L1 = torch.diag(L1 @ L1.t()).reshape(-1,1)
        L2 = torch.diag(L2 @ L2.t()).reshape(-1,1)
        L3 = torch.diag(L3 @ L3.t()).reshape(-1,1)
        L5 = torch.diag(L5 @ L5.t()).reshape(-1,1)
        L6 = torch.diag(L6 @ L6.t()).reshape(-1,1)

        # with torch.no_grad():
            # predbeta = torch.cat((self.w1h.weight,self.w2h.weight))/2
            # predbeta = self.who(predbeta)
            # print(predbeta)

        loss = (L1.t()) @ L1 + (L2.t()) @ L2 +\
               (L3.t()) @ L3 + (L4.t()) @ L4
        # loss +=(L5.t()) @ L5 + (L6.t()) @ L6
        return loss.squeeze()

class OLS(torch.nn.Module):
    def __init__(self,numfeat):
        torch.nn.Module.__init__(self)
        self.linear = torch.nn.Linear(numfeat,1)
    def forward1(self,x1,x2,y):
        return self.forward(x1,x2,y)

    def forward(self,x1,x2,y):
        x = torch.cat((x1,x2),dim=1)

        # with torch.no_grad():
            # betahat = torch.inverse(x.t() @ x) @ x.t() @ y
            # grad = betahat - self.linear.weight.t()
            # print((grad.t() @ grad) / x.shape[0])
            # print(betahat)
        # print(self.linear.weight.detach().numpy())
        loss = self.linear(x)-y
        # pdb.set_trace()
        return (loss.t() @ loss).squeeze()


class Model_Exp6(torch.nn.Module):
    def __init__(self,indim,hiddim,zdim=1,ydim=1):
        torch.nn.Module.__init__(self)
        self.inhid = torch.nn.Linear(indim,hiddim,bias=False)
        self.hidz = torch.nn.Linear(hiddim,zdim)
        self.hidy = torch.nn.Linear(hiddim,ydim)

    def forwardy(self,x,y):
        h = self.inhid(x)
        # h = torch.sigmoid(h)
        yprime = self.hidy(h)
        error = y - yprime
        return (error.t() @ error).trace().squeeze()

    def forwardz(self,x,z,debug=False):
        h = self.inhid(x)
        # h = torch.sigmoid(h)
        zprime = self.hidz(h)
        error = z - zprime
        error = (error.t() @ error).trace().squeeze() 
        if debug: pdb.set_trace()
        return error

def train_model6(model,optimizer,trainxs,labels,predzs=False,finetune=False,numits=10,bsize=10):
    sumloss = 0
    numepochs = 1000 if predzs or finetune else 2*1000
    for epoch in range(numepochs):
        idxs = np.random.rand(numits,bsize) * trainxs.shape[0]
        idxs = idxs.astype(np.int64)
        for start in range(numits):
            optimizer.zero_grad()
            data = trainxs[idxs[start]]
            by = labels[idxs[start]]
            by = by.reshape(-1,labels.shape[1])

            if predzs: loss = model.forwardz(data,by) / bsize
            else: loss = model.forwardy(data,by) / bsize
            sumloss+=loss
            loss.backward()
            optimizer.step()
        print("it: ",epoch,"/",numepochs-1,
          ", avg loss per sample: %.4f" %(sumloss/(epoch+1)/numits/bsize), end="\r" if epoch<numepochs-1 else "\n")

class Deep_Regression(torch.nn.Module):
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

def train_dr(model,optimizer,trainm1,trainm2,labels,bsize=10,verbose=False,early_stop=0.001,numepochs = 200):
    numtrain = trainm1.shape[0]
    numbatches = int(numtrain / bsize)
    for epoch in range(numepochs):
        epochloss = 0
        idxs = (np.random.rand(numbatches,bsize)*numtrain).astype(np.int64)
        # idxs = np.arange(numtrain)
        # np.random.shuffle(idxs)
        # for start in range(numbatches):
        for batch in idxs:
            optimizer.zero_grad()
            m1 = trainm1[batch]
            m2 = trainm2[batch]
            by = labels[batch]
            by = by.reshape(-1,labels.shape[1])

            py = model.forward(m1,m2)
            loss = ((py - by)**2).sum() / bsize / 2
            epochloss+=loss
            # if start==0:
                # print("\npy",py)
                # pdb.set_trace()
                # print("by",by)self.paramsH
            loss.backward()
            # print(epoch,start)
            # print(model.betah_ky.weight)
            # if epoch==0 and start==43: pdb.set_trace()
            # if (model.beta2.weight.grad!=model.beta2.weight.grad).any(): pdb.set_trace()
            optimizer.step()
        avgsampleloss = (epochloss/numbatches/bsize)
        if verbose:
            print("epoch: ",epoch,"/",numepochs-1,
              ", avg loss per sample: %.4f" %avgsampleloss, end="\r" if epoch<numepochs-1 else "\n")
        if avgsampleloss < early_stop: 
            break
