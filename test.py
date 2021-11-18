import torch
from torch.utils.data import DataLoader
from utils.tools import MyDefiniteSampler
from torch.utils.data.sampler import RandomSampler

indices = list(range(10))
sampler = MyDefiniteSampler(indices)
dataset = torch.arange(0, 10).unsqueeze(-1).expand(10, 3)
train_loader = DataLoader(dataset, 1, shuffle=True)

print(train_loader.sampler)
for d in train_loader:
    print(d)


