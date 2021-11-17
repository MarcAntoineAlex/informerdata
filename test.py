import torch
from torch.utils.data import DataLoader
from utils.tools import MyDefiniteSampler

indices = list(range(10))
sampler = MyDefiniteSampler(indices)
dataset = torch.arange(0, 10).unsqueeze(-1).expand(10, 3)
train_loader = DataLoader(dataset, 1, sampler=sampler)
for d in train_loader:
    print(d)

arch = torch.arange(10)[:, None, None]
print(arch)
indice = sampler.indice
print(arch[indice, :, :])

