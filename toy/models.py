import torch
import pdb

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
        loss = torch.trace(L4.t() @ L4)
        return loss

    def forward(self,X1,X2,Y):
        # assumes this is a batch of size greater than 1
        H1 = self.w1h(X1)
        H2 = self.w2h(X2)
        L1 = self.wh2(H1) - X2
        L2 = self.wh1(H2) - X1
        L3 = H1 - H2
        L4 = self.who(H1+H2)*0.5 - Y

        L1 = torch.diag(L1 @ L1.t()).reshape(-1,1)
        L2 = torch.diag(L2 @ L2.t()).reshape(-1,1)
        L3 = torch.diag(L3 @ L3.t()).reshape(-1,1)

        with torch.no_grad():
            predbeta = torch.cat((self.w1h.weight,self.w2h.weight))/2
            predbeta = self.who(predbeta)
            print(predbeta)

        loss = (L1.t()) @ L1 + (L2.t()) @ L2 +\
               (L3.t()) @ L3 + (L4.t()) @ L4
        return loss.squeeze()

class OLS(torch.nn.Module):
    def __init__(self,numfeat):
        torch.nn.Module.__init__(self)
        self.linear = torch.nn.Linear(numfeat,1)
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