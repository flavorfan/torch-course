import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transform

print(torch.__version__)
print(torchvision.__version__)


class Network(nn.Module):
    def __init__(self, channels=1): # default grayscale
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=6, kernel_size=5) 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120) # ((28-5+1)/2 -5 +1)/2 = 4
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, (2, 2), stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, (2, 2), stride=2)

        t = t.reshape(-1, 12*4*4)
        t = F.relu(self.fc1(t))

        t = F.relu(self.fc2(t))

        t = self.out(t)
        
        return t

def get_num_correct(preds, labels):
    return (preds.argmax(dim=1) == labels).sum().item()


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    download=True,
    transform=transform.Compose([
        transform.ToTensor()
    ]))


# ------------------
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100) 
network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)

total_loss = 0
total_correct = 0
for batch in train_loader:
    images, labels = batch 

    preds = network(images) 
    loss = F.cross_entropy(preds, labels) 

    optimizer.zero_grad() 
    loss.backward()  # calculate gradients
    optimizer.step() # update weights using gradients using adam

    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)
    
print(
    "epoch:", 0, 
    "total_correct:", total_correct, 
    "loss:", total_loss
)
