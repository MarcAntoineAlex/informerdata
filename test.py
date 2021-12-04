from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.fft import fft, ifft, irfft
from models.model import Normal


# model = Normal(num=30, length=3000)
# means = torch.linspace(0, 3000, 30, requires_grad=True)
# after = model(means).squeeze().detach().numpy()
# plt.plot(after)
# plt.show()
# data = []
# data_factor = []
#
# for i in range(7):
#     data.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/380987/0/arch{}.npy'.format(i))))
#     data_factor.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/380987/0/arch_factor{}.npy'.format(i))))
# #
# normal = Normal(num=84, length=8521)
# #
# for i in range(1, 9):
#     afteri = normal(data[i], data_factor[i]).squeeze().detach().numpy()
#     afteri_1 = normal(data[i-1], data_factor[i-1]).squeeze().detach().numpy()
#     # diff = afteri - afteri_1
#     # plt.scatter(np.arange(len(afteri_1)), afteri_1, s=0.5)
#     # plt.plot(data_factor[i-1])
#     # plt.show()
#     plt.plot(afteri_1)
#     plt.show()

class Fourrier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sin = nn.Parameter(1 / torch.arange(1, 51).unsqueeze(0)/3)
        self.cos = nn.Parameter(1 / torch.arange(1, 51).unsqueeze(0)/3)

    def forward(self):
        x = torch.arange(1000)[:, None].expand(1000, 50) * 3.1415 / 1000
        x = x * torch.arange(50)[None, :].float()
        sin = torch.sin(x) * self.sin
        cos = torch.cos(x) * self.cos

        return (sin + cos).sum(-1)


f = Fourrier()
f.train()
optim = torch.optim.SGD(f.parameters(), 0.1)
target = torch.ones(1000) * 0.5
for i in range(10):
    optim.zero_grad()
    loss = mse_loss(f(), target)
    loss.backward()
    optim.step()
r = f().detach().numpy()
plt.plot(r)
plt.show()
