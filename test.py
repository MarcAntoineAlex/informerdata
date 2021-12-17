from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.fft import fft, ifft, irfft
from models.model import Fourrier

cos = []
sin = []
for i in range(10):
    cos.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/387240/0/cos{}.npy'.format(i))))
    sin.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/387240/0/sin{}.npy'.format(i))))

for i in range(10):
    print(cos[i][:20])
    f = Fourrier(8521, None, sin[i], cos[i])
    res = f().detach().squeeze().numpy()
    plt.plot(res)
    plt.show()

# x = np.linspace(0, 5, 1000)
# y = 380 * (48.7 * (2288.2 + 33.54 * np.power(x, 2)) ** 0.5 - x * 340) / 10000
#
# plt.plot(x, y)
# plt.xlabel("Velocity/mach")
# plt.ylabel("Thrust/ton")
# plt.show()


