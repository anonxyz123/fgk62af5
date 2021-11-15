import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import sys
import os
from pathlib import Path
lib_dir = (Path(__file__).parent / "lib").resolve()
print(lib_dir)
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

from lib.datasets.DownsampledImageNet import ImageNet16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = ImageNet16(root='/home/lucky/datasets/cifar.python/ImageNet16', train=True,
                      transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), use_num_of_class_only=120)

valset = ImageNet16(root='/home/lucky/datasets/cifar.python/ImageNet16', train=False,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]), use_num_of_class_only=120)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=256,
                                        shuffle=False)
EPOCHS = 200

os.makedirs('./ResNet18_base/', exist_ok=True)
resnet18 = models.resnet50(pretrained=False).to(device)


criterion = nn.CrossEntropyLoss()
resnet18.fc = nn.Sequential(
    nn.Linear(2048, 120),
    #nn.BatchNorm1d(256),
    #nn.ReLU(inplace=True),
    #nn.Dropout(0.5),
    #nn.Linear(256, 100),
)
print(resnet18)
#resnet18.classifier = nn.Sequential(
#    nn.Linear(25088, 256),
#    nn.BatchNorm1d(256),
#    nn.ReLU(inplace=True),
#    nn.Dropout(0.5),
#    nn.Linear(256, 120),
#)
resnet18 = resnet18.to(device)
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)
train_losses = []
train_accs = []
val_losses = []
val_accs = []

pytorch_total_params = sum(p.numel() for p in resnet18.parameters())
print(pytorch_total_params)
for epoch in range(EPOCHS):
    running_loss_train = 0.0
    running_loss_val = 0.0
    resnet18.train()
    correct = 0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss_train += loss.item()
        max_index = outputs.max(dim=1)[1]
        correct += (max_index == labels).sum()
    accuracy = 100 * correct / len(trainset)
    train_accs.append(accuracy)
    print('[%d] train-loss: %.3f, train-acc: %.1f' % (epoch + 1, running_loss_train / len(trainloader), accuracy))
    train_losses.append(running_loss_train / len(trainloader))
    resnet18.eval()
    correct = 0
    for data in valloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        running_loss_val += loss.item()
        max_index = outputs.max(dim=1)[1]
        correct += (max_index == labels).sum()
    accuracy = 100 * correct / len(valset)
    val_accs.append(accuracy)
    print('[%d] val-loss: %.3f, val-acc: %.1f' % (epoch + 1, running_loss_val / len(valloader), accuracy))
    val_losses.append(running_loss_val / len(valloader))
    torch.save(resnet18.cpu(), "./ResNet18_base/resnet18_%d" % (epoch + 1))
    resnet18.to(device)
