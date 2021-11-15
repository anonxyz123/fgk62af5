import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random

trainset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=True,
                                         download=False, transform=transforms.Compose([transforms.ToTensor()]))


def gen_random_samples_classes(ds, fraction, ds_name):
    indices = []
    if ds_name == "CIFAR100":
        for _ in range(100):
            indices.append([])
    for i, data in enumerate(ds):
        indices[data[1]].append(i)

    result = []
    for i, l in enumerate(indices):
        random.shuffle(indices[i])
        result += indices[i][:int(len(indices[i]) * fraction)]

    import pickle
    with open('/home/lucky/datasets/cifar.python/ccrs_' + str(fraction) + '.pkl', 'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == '__main__':
    gen_random_samples_classes(trainset, 0.75, "CIFAR100")
    gen_random_samples_classes(trainset, 0.5, "CIFAR100")
    gen_random_samples_classes(trainset, 0.25, "CIFAR100")
