import torch
import torchvision
import torchvision.transforms as transforms
import random


trainset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=True,
                                         download=False, transform=transforms.Compose([transforms.ToTensor()]))


def gen_random_samples_all(ds, fraction, ds_name):
    indices = list(range(0, len(ds)))
    random.shuffle(indices)
    result = indices[:int(len(ds) * fraction)]

    import pickle
    with open('/home/lucky/datasets/cifar.python/rs_' + str(fraction) + '.pkl', 'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == '__main__':
    gen_random_samples_all(trainset, 0.75, "CIFAR100")
    gen_random_samples_all(trainset, 0.5, "CIFAR100")
    gen_random_samples_all(trainset, 0.25, "CIFAR100")