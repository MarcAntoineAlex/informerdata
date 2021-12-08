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
#
# for i in range(8):
#     data.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/383343/0/arch{}.npy'.format(i))))
#
# for i in range(8):
#     plt.scatter(np.arange(data[i].shape[0]), data[i], s=0.5)
#     plt.show()
class Fourrier(torch.nn.Module):
    def __init__(self, train_length):
        super().__init__()
        self.train_length = train_length
        self.nparam = self.train_length//50
        self.sin = nn.Parameter(1 / torch.arange(1, self.nparam+1).unsqueeze(0)/3)
        self.cos = nn.Parameter(1 / torch.arange(1, self.nparam+1).unsqueeze(0)/3)

    def forward(self):
        x = torch.arange(self.train_length)[:, None].expand(self.train_length, self.nparam) * 3.1415 / self.train_length
        x = x * torch.arange(self.nparam)[None, :].float()
        sin = torch.sin(x) * self.sin
        cos = torch.cos(x) * self.cos

        return torch.sigmoid((sin + cos).sum(-1))[:, None, None] * 2


def get_fourrier(train_length):
    f = Fourrier(train_length)
    r = f().detach().numpy()
    plt.plot(r)
    plt.show()
    f.train()
    optim = torch.optim.SGD(f.parameters(), 0.1)
    target = torch.ones(train_length)
    for i in range(200):
        optim.zero_grad()
        loss = mse_loss(f(), target)
        loss.backward()
        optim.step()
        print(loss)
    r = f().detach().numpy()
    print(f.sin, f.cos)
    plt.plot(r)
    plt.show()
    print(f().shape)
    return f

f = get_fourrier(2000)
