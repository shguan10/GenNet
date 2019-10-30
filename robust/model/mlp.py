import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

import pdb
import numpy as np

EPS=1e-6
GAMMA=.01
JACRATE = .01
WD = 0.005

class Net(nn.Module):
  def __init__(self,numlayers=2):
    super(Net, self).__init__()
    self.numlayers = 2
    self.layers = [nn.Linear(784,784) for _ in range(numlayers-1)]
    self.layers.append(nn.Linear(784,10))
    self.params = nn.ParameterList([])
    for l in self.layers:
      self.params.extend(l.parameters())

  def forward(self, x):
    self.zs = []
    for ind,layer in enumerate(self.layers):
      if ind < self.numlayers-1: x = F.relu(layer(x))
      else: x= F.log_softmax(layer(x),dim=1)
      self.zs.append(x)
    for ind,z in enumerate(self.zs):
      z.requires_grad_()
      z.retain_grad()
    return x

def zerograd(model):
  for p in model.parameters():
    p.grad *= 0
  for z in model.zs:
    z.grad -= 0

def trainFCN(args, model, device, train_loader, optimizer, epoch):
  model.train()
  # for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
  for (data, target) in tqdm((train_loader)):
    data, target = data.to(device), target.to(device)
    data.requires_grad_()
    bsize = data.shape[0]
    data = data.reshape((bsize,784))
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward(retain_graph=True)
    # if JACRATE > 0:
    lgs = [p.grad.clone().detach() for p in model.parameters()]
    zerograd(model)
    pi = (torch.rand(bsize)*10).long()
    mask = torch.zeros(output.shape)
    for b in range(bsize):
      mask[b,pi[b]] = 1
    mask = mask.to(device)
    output.backward(mask)
    with torch.no_grad():
      for l in range(model.numlayers):
        bshape = model.layers[l].bias.shape[0]
        tmp = model.zs[l].grad.t() @ model.zs[l].grad
        # pdb.set_trace()
        tmp = tmp @ model.layers[l].weight
        lgs[l*2] += (JACRATE/bsize)*2 * tmp

    for p,g in zip(model.parameters(),lgs):
      p.grad = g
    optimizer.step()
    # if batch_idx % 100==0: print(loss.item())
    # if batch_idx % args.log_interval == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader,val=True):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      bsize = data.shape[0]
      data = data.reshape((bsize,784))
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  if val:
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
  else:
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))


def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=120, metavar='N',
            help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
  
  parser.add_argument('--save-model', action='store_true', default=False,
            help='For Saving the current Model')

  parser.add_argument('--weight-decay', action='store_true', default=False, 
            help='use weight decay')
  parser.add_argument('--numlayers', type=int, default=2, metavar='ls',
            help='(default: 2)')
  args = parser.parse_args()
  args.large_margin = not args.weight_decay
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  
  imtransform = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                 ])

  train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../../data', train=True, download=True,
                        transform=imtransform),
                    batch_size=args.batch_size, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../../data', train=False, download=True,
                        transform=imtransform),
                    batch_size=args.test_batch_size, shuffle=False,**kwargs)
  
  val = torch.load("../../data/MNIST/processed/validate.pt")
  val = list(val)
  val[0] = [imtransform(i.numpy()) for i in val[0]]
  val = list(zip(val[0],val[1]))
  val_loader = torch.utils.data.DataLoader(val,batch_size=args.test_batch_size,shuffle=False,**kwargs)

  model = Net(args.numlayers).to(device)
  optimizer = optim.SGD(model.parameters(), 
                        lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=WD if args.weight_decay else 0)

  for epoch in range(1, args.epochs + 1):
    if epoch % 40 ==0:
      print("decayed lr")
      for param_group in optimizer.param_groups:
        param_group['lr'] /= 10
    if args.large_margin:
      trainFCN(args, model, device, train_loader, optimizer, epoch)
    else: 
      train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, val_loader)
  test(args, model, device, test_loader,val=False)

  if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
  main()
  # testJ()