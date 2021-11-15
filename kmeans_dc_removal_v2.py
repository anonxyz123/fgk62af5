import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import sys
sys.path.insert(0, './deepcluster-master')
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(
         (0.485, 0.456, 0.406),
         (0.229, 0.224, 0.225)
     )])

trainset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=True,
                                         download=True, transform=transform)
valset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=False,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                        shuffle=False)


def k_means_removal_dc(ds, fraction, n_clusters, n_init=100, max_iter=300):
    from vgg import vgg16
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pickle
    print("Konverting to numpy arrays...")
    loading_file = torch.load("/home/lucky/dc_exps/100/checkpoint.pth.tar")

    model = loading_file['model'].to(device)
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    features = []
    with torch.no_grad():
        for i, data in enumerate(ds):
            features.append(model(data[0].to(device))[0].cpu().numpy())
    features = np.array(features)

    pca = PCA(n_components=64)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pcomponents = pca.fit_transform(scaled_features)
    # print(pcomponents.shape)
    print("Starting scaling and KMeans...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pcomponents)
    kmeans = KMeans(init="random",
                    n_clusters=n_clusters,
                    n_init=n_init,
                    max_iter=max_iter)
    kmeans.fit(scaled_features)
    print("Finished K-Means...")
    #
    print("Calculating differences...")
    diffs = []
    labels = []
    for i in range(len(pcomponents)):
        labels.append(kmeans.labels_[i])
        diffs.append(np.linalg.norm(kmeans.cluster_centers_[kmeans.labels_[i]] - pcomponents[i]))
    diffs = np.array(diffs)
    for n in range(50):
        sum = 0
        for i in range(len(labels)):
            if labels[i] == n:
                sum += 1
        print(n, sum)
    result = diffs.argsort()[:int(len(diffs) * fraction)].tolist()
    #
    import pickle
    with open('/home/lucky/datasets/cifar.python/kmeans_dc_' + str(n_clusters) + '_removal_' + str(fraction) + '_v2.pkl',
              'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == '__main__':
    k_means_removal_dc(trainloader, 0.75, 100)
    k_means_removal_dc(trainloader, 0.50, 100)
    k_means_removal_dc(trainloader, 0.25, 100)
