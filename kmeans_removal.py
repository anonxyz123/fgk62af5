import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

trainset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=True,
                                         download=False, transform=transforms.Compose([transforms.ToTensor()]))


def k_means_removal(ds, fraction, n_clusters, n_init=100, max_iter=300):
   images = []
   print("Konverting to numpy arrays...")
   for i, data in enumerate(ds):
       images.append(data[0].numpy().reshape((-1)))
   images = np.array(images)
   print(images.shape)
   print("Starting scaling...")
   scaler = StandardScaler()
   scaled_features = scaler.fit_transform(images)

   print("Starting KMeans-" + str(n_clusters) + "...")
   kmeans = KMeans(init="random",
                   n_clusters=n_clusters,
                   n_init=n_init,
                   max_iter=max_iter)
   kmeans.fit(scaled_features)
   print("Finished K-Means...")

   print("Calculating differences...")
   diffs = []
   for i in range(len(scaled_features)):
       diffs.append(np.linalg.norm(kmeans.cluster_centers_[kmeans.labels_[i]] - scaled_features[i]))
   diffs = np.array(diffs)
   result = diffs.argsort()[:int(len(diffs) * fraction)].tolist()

   import pickle
   with open('/home/lucky/datasets/cifar.python/kmeans_' + str(n_clusters) + '_removal_' + str(fraction) + '.pkl', 'wb') as f:
       pickle.dump(result, f)

   return result


if __name__ == '__main__':
    k_means_removal(trainset, 0.75, 50)
    k_means_removal(trainset, 0.50, 50)
    k_means_removal(trainset, 0.25, 50)
    k_means_removal(trainset, 0.75, 100)
    k_means_removal(trainset, 0.50, 100)
    k_means_removal(trainset, 0.25, 100)
    k_means_removal(trainset, 0.75, 150)
    k_means_removal(trainset, 0.50, 150)
    k_means_removal(trainset, 0.25, 150)
    k_means_removal(trainset, 0.75, 200)
    k_means_removal(trainset, 0.50, 200)
    k_means_removal(trainset, 0.25, 200)
