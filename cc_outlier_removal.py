import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np

trainset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=True,
                                         download=False, transform=transforms.Compose([transforms.ToTensor()]))


def cc_outlier_removal(ds, fraction):
   indices = []
   images = []
   means = [0] * 100
   diffs = []
   for _ in range(100):
       indices.append([])
       images.append([])
       diffs.append([])
   for i, data in enumerate(ds):
       indices[data[1]].append(i)
       images[data[1]].append(data[0])

   for i in range(100):
       for j in range(len(images[i])):
           means[i] += images[i][j]
       means[i] /= len(images[i])

   result = []
   for i in range(100):
       for j in range(len(images[i])):
           diffs[i].append(torch.norm(means[i] - images[i][j]))
       diff_np = np.array(diffs[i])
       index_list = diff_np.argsort()[:int(len(images[i])*fraction)].tolist()
       for j in index_list:
           result.append(indices[i][j])

   import pickle
   with open('/home/lucky/datasets/cifar.python/cc_outlier_removal_' + str(fraction) + '.pkl', 'wb') as f:
       pickle.dump(result, f)

   return result


if __name__ == '__main__':
    cc_outlier_removal(trainset, 0.75)
    cc_outlier_removal(trainset, 0.5)
    cc_outlier_removal(trainset, 0.25)
