import torch
import torchvision
import torchvision.transforms as transform
import torch.nn as nn

from torch.utils.data import DataLoader


mean = 0.2860347330570221
std = 0.3530242443084717

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean, std)
    ])
)

loader = DataLoader(train_set, batch_size=100)
images, labels = next(iter(loader))

print(images.shape, labels.shape)

