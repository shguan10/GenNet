import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

import pdb
import numpy as np

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = FoldedConv2d(1, 20, [5,5],pre=False)
    # self.conv1 = nn.Conv2d(1, 20, [5,5])
    self.conv2 = FoldedConv2d(20, 50, [5,5])
    # self.conv2 = nn.Conv2d(20, 50, [5,5])
    self.fc1 = nn.Linear(4*4*50, 500)
    self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 4*4*50)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

  def __foldedmax_pool2d(self,x,k,s,):
    x = x.transpose(-1,-2).contiguous()
    x = x.view(-1,)
    return F.max_pool2d(x,k,s)

class InputLMLoss(torch.Function):

  @staticmethod
  def forward(ctx, input, target,gradqnorm):
    # input and target are shape (batch,logits)
    # gradqnorm is scalar
    bsize,nlogits = input.shape
    for b in range(bsize):
      logits = input[b]
      desired = target[b] #one-hot
      # grad_wrt_x = 
      # right now just do sum
      torch.sum()

    ctx.save_for_backward(input, target)
    output = input.mm(weight.t())
    if bias is not None:
        output += bias.unsqueeze(0).expand_as(output)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    # This is a pattern that is very convenient - at the top of backward
    # unpack saved_tensors and initialize all gradients w.r.t. inputs to
    # None. Thanks to the fact that additional trailing Nones are
    # ignored, the return statement is simple even when the function has
    # optional inputs.
    input, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None

    # These needs_input_grad checks are optional and there only to
    # improve efficiency. If you want to make your code simpler, you can
    # skip them. Returning gradients for inputs that don't require it is
    # not an error.
    if ctx.needs_input_grad[0]:
        grad_input = grad_output.mm(weight)
    if ctx.needs_input_grad[1]:
        grad_weight = grad_output.t().mm(input)
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum(0).squeeze(0)

    return grad_input, grad_weight, grad_bias

def im2col(x,kshape,stride):
  """
  Assumes x is shape (...,inch,row,col)
  output shape (..., newRow*newCols, kshape[0]*kshape[1]*inch)
  """
  inch,rows,cols = x.shape[-3:]

  xshape = x.shape
  x = x.transpose(-1,-3)
  x = x.unfold(len(xshape)-2, kshape[1],stride[1])
  x = x.unfold(len(xshape)-3,kshape[0],stride[0])
  x = x.contiguous().view(*xshape[:-3],-1,
                          kshape[0]*kshape[1]*inch)
  return x

def col2im(x,nRows,nCols):
  """
  Assumes x is shape (..., nRows*nCols, inch)
  output shape (...,inch,nRows,nCols)
  """
  xshape = x.shape
  x = x.transpose(-1,-2).contiguous()
  x = x.view(*xshape[:-2],-1,nRows,nCols)
  return x

class FoldedConv2d(nn.Module):
  def __init__(self,inch,outch,kshape,stride=[1,1]):
    nn.Module.__init__(self)
    k=1
    for j in kshape: k*=j
    self.kernel = nn.parameter.Parameter(torch.Tensor(inch*k,outch))
    self.kshape = kshape
    self.stride = stride
    self.reset_parameters()
    
  def reset_parameters(self):
    torch.nn.init.kaiming_uniform_(self.kernel, a=np.sqrt(5))
    
  def forward(self,x,preneed=True,postneed=True):
    xshape = x.shape
    if preneed:
      x = im2col(x,self.kshape,self.stride)
    x = x @ self.kernel
    if postneed:
      nRows = int((xshape[-2]-(self.kshape[0]-1)-1)/self.stride[0]) +1
      nCols = int((xshape[-1]-(self.kshape[1]-1)-1)/self.stride[1]) +1
      x = col2im(x,nRows,nCols)
    return x

def get_jacobian(net, x, noutputs):
  x = x.repeat(noutputs, 1)
  x.requires_grad_(True)
  y = net.forward(x)
  y.backward(torch.eye(noutputs))
  return x.grad.data

def train(args, model, device, train_loader, optimizer, epoch):
  model.train()
  # for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
  for (data, target) in tqdm((train_loader)):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    # if batch_idx % args.log_interval == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     epoch, batch_idx * len(data), len(train_loader.dataset),
    #     100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def main():
  # Training settings
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument('--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
  parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
  parser.add_argument('--epochs', type=int, default=10, metavar='N',
            help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
            help='learning rate (default: 0.01)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
            help='SGD momentum (default: 0.5)')
  parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
  
  parser.add_argument('--save-model', action='store_true', default=False,
            help='For Saving the current Model')
  args = parser.parse_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
             transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
             ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=False, transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
             ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


  model = Net().to(device)
  optimizer = optim.SGD(model.parameters(), 
                        lr=args.lr, 
                        momentum=args.momentum)

  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)

  if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")

def testJ():
  use_cuda = False

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data', train=True, download=True,
             transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
             ])),
    batch_size=32, shuffle=True, **kwargs)

  model = Net().to(device)
  model.train()
  for (data, target) in tqdm((train_loader)):
    data, target = data.to(device), target.to(device)
    data.requires_grad_()
    pdb.set_trace()
    output = get_jacobian(model,data,10)
    print(output)  

if __name__ == '__main__':
  # main()
  testJ()