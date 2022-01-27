import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np

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

mse = [
    [0.40, 0.501, 0.829, 1.04, 1.182],
    [0.42, 0.492, 0.801, 1.01, 1.118],
    [0.45, 0.478, 0.770, 1.07, 1.131],
    [0.43, 0.479, 0.797, 1.10, 1.094],
    [0.46, 0.523, 0.843, 1.06, 1.073]
]

wd = [0.01, 0.0032, 0.001, 0.00032, 0.0001]
mse = np.array(mse)
best_case = np.array([0.40, 0.478, 0.770, 1.01, 1.073])
plt.figure(1, figsize=(8, 6))
l0, = plt.plot(mse[0], 'h-')
l1, = plt.plot(mse[1], 'o-')
l2, = plt.plot(mse[2], '^-')
l3, = plt.plot(mse[3], 'v-')
l4, = plt.plot(mse[4], 'D-')
plt.xlabel('Prediction length', fontsize=font_size, family=font_type)
plt.ylabel('MSE score', fontsize=font_size, family=font_type)
plt.ylim(0.3, 1.3)
plt.legend(handles=[l0, l1, l2, l3, l4],
           labels=['w_decay=1e-2', 'w_decay=3e-3', 'w_decay=1e-3', 'w_decay=3e-4', 'w_decay=1e-4', ],
           loc=0, prop=font2)
plt.xticks([0, 1, 2, 3, 4], [24, 48, 168, 336, 720], fontsize=font_size-5, rotation=1)
plt.yticks([0.4, 0.6, 0.8, 1, 1.2], fontsize=font_size-5)

plt.savefig("/Users/marc-antoine/Documents/icml/wc.jpg")
plt.show()

# fourier divider
mse = [
    [0.48, 0.534, 0.833, 1.04, 1.073],
    [0.45, 0.482, 0.787, 1.01, 1.123],
    [0.46, 0.510, 0.841, 1.12, 1.192],
    [0.47, 0.478, 0.770, 1.09, 1.151],
    [0.40, 0.532, 0.812, 1.10, 1.146],
]
mse = np.array(mse)
plt.figure(1, figsize=(8, 6))
l0, = plt.plot(mse[0], 'h-')
l1, = plt.plot(mse[1], 'o-')
l2, = plt.plot(mse[2], '^-')
l3, = plt.plot(mse[3], 'v-')
l4, = plt.plot(mse[4], 'D-')
plt.xlabel('Prediction length', fontsize=font_size, family=font_type)
plt.ylabel('MSE score', fontsize=font_size, family=font_type)
plt.ylim(0.3, 1.3)
plt.legend(handles=[l0, l1, l2, l3, l4],
           labels=['d=1', 'd=4', 'd=16', 'd=64', 'd=128'],
           loc=0, prop=font2)
plt.xticks([0, 1, 2, 3, 4], [24, 48, 168, 336, 720], fontsize=font_size-5, rotation=1)
plt.yticks([0.4, 0.6, 0.8, 1, 1.2], fontsize=font_size-5)

plt.savefig("/Users/marc-antoine/Documents/icml/fd.jpg")
plt.show()

# Sigmoid Temperature
mse = [
    [0.42, 0.53, 0.78, 1.11, 1.17],
    [0.40, 0.478, 0.79, 1.09, 1.12],
    [0.41, 0.48, 0.77, 1.04, 1.13],
    [0.41, 0.50, 0.80, 1.01, 1.10],
    [0.44, 0.51, 0.82, 1.02, 1.073],
]
mse = np.array(mse)
plt.figure(1, figsize=(8, 6))
l0, = plt.plot(mse[0], 'h-')
l1, = plt.plot(mse[1], 'o-')
l2, = plt.plot(mse[2], '^-')
l3, = plt.plot(mse[3], 'v-')
l4, = plt.plot(mse[4], 'D-')
plt.xlabel('Prediction length', fontsize=font_size, family=font_type)
plt.ylabel('MSE score', fontsize=font_size, family=font_type)
plt.ylim(0.3, 1.3)
plt.legend(handles=[l0, l1, l2, l3, l4],
           labels=['temperature=1', 'temperature=2', 'temperature=3', 'temperature=4', 'temperature=5'],
           loc=0, prop=font2)
plt.xticks([0, 1, 2, 3, 4], [24, 48, 168, 336, 720], fontsize=font_size-5, rotation=1)
plt.yticks([0.4, 0.6, 0.8, 1, 1.2], fontsize=font_size-5)

plt.savefig("/Users/marc-antoine/Documents/icml/tp.jpg")
plt.show()

