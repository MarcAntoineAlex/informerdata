from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.fft import fft, ifft, irfft
from models.model import Fourrier, sigtemp

plt.rcParams["font.family"] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

font_size = 20
font_type = 'Arial'

font1 = {'family': font_type,
         'weight': 'normal',
         'size': font_size,
         }

font2 = {'family': font_type,
         'weight': 'normal',
         'size': 15,
         }

cos = []
sin = []
da, dw = [], []
job = "406602"
num = len(os.listdir("/Users/marc-antoine/Desktop/{}/0".format(job)))//4
for i in range(num):
    cos.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/{}/0/cos{}.npy'.format(job, i))))
    sin.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/{}/0/sin{}.npy'.format(job, i))))
    da.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/{}/0/da{}.npy'.format(job, i+1))))
    dw.append(torch.from_numpy(np.load('/Users/marc-antoine/Desktop/{}/0/dw{}.npy'.format(job, i+1))))

for i in range(num):
    print(cos[i])
    f = Fourrier(8360, 40, None, sin[i], cos[i])
    res = sigtemp(f(), 1).detach().squeeze().numpy()
    # res = f().detach().squeeze().numpy()
    plt.figure(1, figsize=(10, 6))
    plt.plot(res, label='P', color='black')
    plt.legend(prop=font2)
    plt.ylim(0.2, 1)
    plt.xlabel('$a_j$', fontsize=font_size, family=font_type)
    plt.ylabel('$p_i(A)$', fontsize=font_size, family=font_type)
    plt.xticks(fontsize=font_size - 5, rotation=1)
    plt.yticks(fontsize=font_size - 5)
    plt.savefig("/Users/marc-antoine/Documents/icml/P(A){}.pdf".format(i), dpi=300, bbox_inches='tight')
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

# class M(nn.Module):
#     def __init__(self):
#         super(M, self).__init__()
#         self.l = nn.Linear(3, 4)
#     def forward(self, x):
#         return self.l(x)
#
# m = M()
# x = torch.randn(3, 3)
# m.l = nn.Linear()