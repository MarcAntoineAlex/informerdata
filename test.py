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
    cos.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/387890/0/cos{}.npy'.format(i))))
    sin.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/387890/0/sin{}.npy'.format(i))))

for i in range(10):
    print(cos[i][:20])
    f = Fourrier(8521, None, sin[i], cos[i])
    res = f().detach().squeeze().numpy()
    plt.plot(res)
    plt.show()

# Ma = np.linspace(0, 5, 1000)
# Mair = 320 + 16 * Ma
# F = 6740.2 + 72.12 * np.power(Ma, 2) - 512847.4 / Mair
# F = np.power(F * 1000, 0.5)
# F = 33/32 * Mair * F
# print(F)
# print(Mair * Ma * 340)
# F = F - Mair * Ma * 340
#
# P = F * Ma * 340
# Q = 143000000 * Mair / 32
#
# efficiency = P / Q
#
# plt.plot(Ma, F/10000)
# plt.xlabel("Velocity/mach")
# plt.ylabel("Thrust/ton")
# plt.show()
#
# plt.plot(Ma, efficiency)
# plt.show()


