import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import sys, os
import random
import numpy as np
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


trainset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=True,
                                         download=False, transform=transforms.Compose([transforms.ToTensor()]))

valset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=False,
                                         download=False, transform=transforms.Compose([transforms.ToTensor()]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)
trainloader_iter1 = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                          shuffle=False)


def transfer_learning_resnet18(epochs, train_loader, val_loader):
    os.makedirs('./outputTLResnet18/', exist_ok=True)
    resnet18 = models.resnet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    for param in resnet18.parameters():
        param.requires_grad = False

    resnet18.fc.requires_grad_(True)
    resnet18.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 100),
    )
    resnet18 = resnet18.to(device)
    optimizer = optim.SGD(resnet18.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=3e-4)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss_train = 0.0
        running_loss_val = 0.0
        pbar = tqdm(train_loader, leave=False,
                    file=sys.stdout, ascii=True)

        resnet18.train()
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()

        print('[%d] train-loss: %.3f' % (epoch + 1, running_loss_train / len(train_loader)))
        train_losses.append(running_loss_train / len(train_loader))

        resnet18.eval()
        pbar = tqdm(val_loader, leave=False,
                    file=sys.stdout, ascii=True)
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            running_loss_val += loss.item()

        print('[%d] val-loss: %.3f' % (epoch + 1, running_loss_val / len(val_loader)))
        val_losses.append(running_loss_val / len(val_loader))

        torch.save(resnet18.cpu(), "./outputTLResnet18/resnet18_%d" % (epoch + 1))
        resnet18.to(device)

    with open('./outputTLResnet18/train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open('./outputTLResnet18/val_losses.pkl', 'wb') as f:
        pickle.dump(val_losses, f)

    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(loc='lower left')
    plt.savefig("./outputTLResnet18/loss_curves.pdf")


def TLResNetIndices(epoch, train_loader, fraction, ascending=True):
    model = torch.load("./outputTLResnet18/resnet18_%d" % (epoch))

    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, leave=False,
                file=sys.stdout, ascii=True)

    model.to(device).eval()
    losses = []
    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
    losses = np.array(losses)

    if ascending:
        result = losses.argsort()[:int(len(train_loader.dataset)*fraction)].tolist()
        with open('/home/lucky/datasets/cifar.python/TLResNet18_easy_' + str(fraction) + '.pkl', 'wb') as f:
            pickle.dump(result, f)
    else:
        result = losses.argsort()[::-1][:int(len(train_loader.dataset) * fraction)].tolist()
        with open('/home/lucky/datasets/cifar.python/TLResNet18_hard_' + str(fraction) + '.pkl', 'wb') as f:
            pickle.dump(result, f)


if __name__ == '__main__':
    transfer_learning_resnet18(epochs=75, train_loader=trainloader, val_loader=valloader)
    TLResNetIndices(50, trainloader_iter1, 0.75, True)
    TLResNetIndices(50, trainloader_iter1, 0.50, True)
    TLResNetIndices(50, trainloader_iter1, 0.25, True)
    TLResNetIndices(50, trainloader_iter1, 0.75, False)
    TLResNetIndices(50, trainloader_iter1, 0.50, False)
    TLResNetIndices(50, trainloader_iter1, 0.25, False)
