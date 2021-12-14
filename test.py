from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.fft import fft, ifft, irfft
from models.model import Fourrier


# model = Normal(num=30, length=3000)
# means = torch.linspace(0, 3000, 30, requires_grad=True)
# after = model(means).squeeze().detach().numpy()
# plt.plot(after)
# plt.show()

cos = []
sin = []
for i in range(10):
    cos.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/386768/0/cos{}.npy'.format(i))))
    sin.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/386768/0/sin{}.npy'.format(i))))

f = Fourrier(8521)
for i in range(10):
    print(sin[i])
    f = Fourrier(8521, None, sin[i], cos[i])
    res = f().detach().squeeze().numpy()
    plt.plot(res[:100])
    print(res)
    plt.show()


