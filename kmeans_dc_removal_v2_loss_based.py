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


def k_means_removal_dc(ds, fraction):
    from vgg import vgg16
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pickle
    #print("Konverting to numpy arrays...")
    loading_file = torch.load("/home/lucky/dc_exps/100/checkpoint.pth.tar")
    with open('/home/lucky/dc_exps/100/clusters', 'rb') as f:
        clusters = pickle.load(f)[-1]

    image_to_cluster = [-1] * len(ds)
    for c in range(len(clusters)):
        for e in clusters[c]:
            image_to_cluster[e] = c

    model = loading_file['model'].to(device)
    for param in model.parameters():
        param.requires_grad = False
    # set last fully connected layer
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Linear(256, 1000)
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()
    model.top_layer.cuda()

    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.top_layer.parameters():
        param.requires_grad = True


    mean_cluster_centers = []
    for c in range(len(clusters)):
        mean_cluster_centers.append(np.mean(features[clusters[c]]))

    # print(pcomponents.shape)

    diffs = []
    labels = []
    for i in range(len(ds)):
        diffs.append(np.linalg.norm(mean_cluster_centers[image_to_cluster[i]] - features[i]))
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
    with open('/home/lucky/datasets/cifar.python/kmeans_dc_' + str(n_clusters) + '_removal_' + str(fraction) + '_v2_loss_based.pkl',
              'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == '__main__':
    k_means_removal_dc(trainloader, 0.75)
    k_means_removal_dc(trainloader, 0.50)
    k_means_removal_dc(trainloader, 0.25)
