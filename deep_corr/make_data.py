import torchvision
import pdb

import pickle as pk

import numpy as np

def make_mnist_dataset():
  train_mnist = torchvision.datasets.MNIST(
    root="../data",train=True, 
    download=True, transform=torchvision.transforms.ToTensor()
  )
  indicies = np.random.permutation(len(train_mnist))
  train_x = train_mnist.data[indicies].reshape(-1,28*28).float().numpy()
  train_y = train_mnist.targets[indicies].numpy()

  test_mnist = torchvision.datasets.MNIST(
    root="../data",train=False, 
    download=True, transform=torchvision.transforms.ToTensor()
  )

  indicies = np.random.permutation(len(test_mnist))
  test_x = test_mnist.data[indicies].reshape(-1,28*28).float().numpy()
  test_y = test_mnist.targets[indicies].numpy()
  

  mu = train_x.mean(axis=0)
  sigma = train_x.std(axis=0)

  sigma = np.array([s if s>0 else 1 for s in sigma])

  train_x = (train_x-mu)/sigma
  test_x = (test_x-mu)/sigma

  # pdb.set_trace()
  with open("mnist_shuffled.pk","wb") as f: pk.dump(((train_x,train_y),(test_x,test_y)),f)

  totalcols = train_x.shape[1]

  for i in range(5):
    percent = (i+1)*10
    # the left i% of the image is m1, the rest is m2
    m1_cols = int(totalcols*percent)
    m2_cols = totalcols - m1_cols
    train_m1,test_m1 = train_x[:,:m1_cols],test_x[:,:m1_cols]
    train_m2,test_m2 = train_x[:,m1_cols:],test_x[:,m1_cols:]

    with open("mnist_split"+str(percent)+".pk","wb") as f: pk.dump((((train_m1,train_m2),train_y),((test_m1,test_m2),test_y)),f)

if __name__ == '__main__':
  make_mnist_dataset()