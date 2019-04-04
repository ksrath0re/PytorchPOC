import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
#Interface to use data transformation

train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)

torch.set_printoptions(linewidth=120)

print(len(train_set))
print(train_set.targets) #train_lables has been made deprecated
print(train_set.targets.bincount())

sample = next(iter(train_set))
image, target = sample

print(image.reshape(1,28,28))
print(torch.tensor(target).shape)

plt.imshow(image.squeeze(), cmap='gray')
plt.show() #Need to add because otherwise it won't show plot in PYCHARM
print('Label :', target)