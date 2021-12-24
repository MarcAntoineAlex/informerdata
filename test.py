from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.fft import fft, ifft, irfft
from models.model import Fourrier, sigtemp

cos = []
sin = []
job = "391924"
num = len(os.listdir("/Users/marc-antoine/Desktop/{}/0".format(job)))//2
for i in range(num):
    cos.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/{}/0/cos{}.npy'.format(job, i))))
    sin.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/{}/0/sin{}.npy'.format(job, i))))

for i in range(num):
    print(cos[i][:10])
    f = Fourrier(8377, None, sin[i], cos[i])
    res = 2 * sigtemp(f(), 0.5).detach().squeeze().numpy()
    plt.plot(res)
    plt.ylim(0, 2)
    plt.show()

# Ma = np.linspace(0, 5, 1000)
# Mair = 320 + 16 * Ma
# F = 1264.9 * np.power(Ma, 3) + 25301.7 * np.power(Ma, 2) + 32330.1 * Ma - 606806.2
# F = 6486.9 - F/Mair + 99552.5/(7.5*Ma+150)
# F = np.power(F * 1000, 0.5)
# F = 33/32 * Mair * F
# F = F - Mair * Ma * 340
#
# P = F * Ma * 340
# Q = 143000000 * Mair / 32
#
# efficiency = P / Q
#
# plt.plot(Ma, F/10000)
#
# plt.xlabel("Velocity/mach")
# plt.ylabel("Thrust/ton")
# plt.grid()
# plt.xlim(0, 5)
# plt.savefig("/Users/marc-antoine/Desktop/F.jpg")
# plt.show()
#
# plt.plot(Ma, efficiency)
# plt.xlabel("Velocity/mach")
# plt.ylabel("Efficiency")
# plt.xlim(0, 5)
# plt.ylim(0, 1)
# plt.grid()
# plt.savefig("/Users/marc-antoine/Desktop/Efficiency.jpg")
# plt.show()
#
# print(F[600]/10000)
# print(F[0], F[-1])
# print(efficiency[600])