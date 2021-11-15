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
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=True,
                                         download=True, transform=transform)
valset = torchvision.datasets.CIFAR100(root='/home/lucky/datasets/cifar.python/', train=False,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=128,
                                        shuffle=False)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),   # [batch, 8, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # [batch, 16, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [batch, 32, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),  # [batch, 64, 2, 2]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # [batch, 32, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [batch, 16, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # [batch, 8, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def autoencoder_training(epochs, train_set, val_set):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                              shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128,
                                              shuffle=False)
    os.makedirs('./outputAE/', exist_ok=True)
    model = Autoencoder()
    criterion = nn.BCELoss()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss_train = 0.0
        running_loss_val = 0.0
        pbar = tqdm(train_loader, leave=False,
                    file=sys.stdout, ascii=True)

        model.train()
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)

            optimizer.zero_grad()

            output_encoded, output_decoded = model(inputs)
            loss = criterion(output_decoded, inputs)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()

        print('[%d] train-loss: %.3f' % (epoch + 1, running_loss_train / len(train_loader)))
        train_losses.append(running_loss_train / len(train_loader))

        model.eval()
        pbar = tqdm(val_loader, leave=False,
                    file=sys.stdout, ascii=True)
        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)

            output_encoded, output_decoded = model(inputs)
            loss = criterion(output_decoded, inputs)
            running_loss_val += loss.item()

        print('[%d] val-loss: %.3f' % (epoch + 1, running_loss_val / len(val_loader)))
        val_losses.append(running_loss_val / len(val_loader))

        torch.save(model.cpu(), "./outputAE/ae_%d" % (epoch + 1))
        model.to(device)

        vutils.save_image(output_decoded[:64], './outputAE/fake_%d.pdf' % (epoch + 1))
        if epoch == 0:
            vutils.save_image(inputs[:64], './outputAE/real.pdf')

    import pickle
    with open('./outputAE/train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open('./outputAE/val_losses.pkl', 'wb') as f:
        pickle.dump(val_losses, f)

    import matplotlib.pyplot as plt
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(loc='lower left')
    plt.savefig("./outputAE/loss_curves.pdf")
    plt.show()


def k_means_removal_ae(epoch, ds, fraction, n_clusters, n_init=100, max_iter=300):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    images = []
    print("Konverting to numpy arrays...")
    model = torch.load("./outputAE/ae_%d" % (epoch))
    model.eval()
    for i, data in enumerate(ds):
        output_encoded, output_decoded = model(data[0])
        images.append(output_encoded.detach().numpy().reshape((-1)))
    images = np.array(images)
    print(images.shape)
    print("Starting scaling...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(images)

    print("Starting KMeans...")
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
    with open('/home/lucky/datasets/cifar.python/kmeans_ae_' + str(n_clusters) + '_removal_' + str(fraction) + '.pkl', 'wb') as f:
        pickle.dump(result, f)

    return result


if __name__ == '__main__':
    autoencoder_training(50, trainset, valset)
    k_means_removal_ae(50, trainloader, 0.75, 50)
    k_means_removal_ae(50, trainloader, 0.50, 50)
    k_means_removal_ae(50, trainloader, 0.25, 50)
    k_means_removal_ae(50, trainloader, 0.75, 100)
    k_means_removal_ae(50, trainloader, 0.50, 100)
    k_means_removal_ae(50, trainloader, 0.25, 100)
    k_means_removal_ae(50, trainloader, 0.75, 150)
    k_means_removal_ae(50, trainloader, 0.50, 150)
    k_means_removal_ae(50, trainloader, 0.25, 150)
    k_means_removal_ae(50, trainloader, 0.75, 200)
    k_means_removal_ae(50, trainloader, 0.50, 200)
    k_means_removal_ae(50, trainloader, 0.25, 200)
