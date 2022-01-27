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
best_case = np.array([0.40, 0.478, 0.770, 1.01, 1.073])
baseline = np.array([0.577, 0.685, 0.931, 1.128, 1.215])

# Weight Decay
mse = [
    [0.46, 0.523, 0.843, 1.02, 1.073],
    [0.43, 0.492, 0.797, 1.01, 1.094],
    [0.45, 0.488, 0.770, 1.04, 1.131],
    [0.42, 0.479, 0.801, 1.07, 1.118],
    [0.40, 0.501, 0.829, 1.09, 1.182]
]

mse = np.array(mse)
# mse = mse - best_case
mse = -mse + baseline
mse = mse.T

plt.figure(1, figsize=(8, 7))
l0, = plt.plot(mse[0], 'h-', color='red')
l1, = plt.plot(mse[1], 'o-', color='orange')
l2, = plt.plot(mse[2], '^-', color='green')
l3, = plt.plot(mse[3], 'v-', color='purple')
l4, = plt.plot(mse[4], 'D-', color='blue')
plt.xlabel('Weight decay '+'$d_W$', fontsize=font_size, family=font_type)
plt.ylabel('MSE score improvement', fontsize=font_size, family=font_type)
plt.ylim(-0.02, 0.23)
plt.legend(handles=[l0, l1, l2, l3, l4],
           labels=['pred_len=24', 'pred_len=48', 'pred_len=168', 'pred_len=336', 'pred_len=720'],
           loc=0, prop=font2)
plt.xticks([0, 1, 2, 3, 4], ['$10^{-4}$', '$10^{-3.5}$', '$10^{-3}$', '$10^{-2.5}$', '$10^{-2}$'], fontsize=font_size-5, rotation=1)
plt.yticks([0, 0.05, 0.10, 0.15, 0.20], fontsize=font_size-5)

plt.savefig("/Users/marc-antoine/Documents/icml/weight_decay.pdf", dpi=300, bbox_inches='tight')
plt.show()

# fourier divider
mse = [
    [0.48, 0.534, 0.833, 1.04, 1.073],
    [0.45, 0.546, 0.787, 1.01, 1.123],
    [0.46, 0.510, 0.841, 1.12, 1.192],
    [0.47, 0.478, 0.770, 1.09, 1.151],
    [0.40, 0.532, 0.812, 1.10, 1.146],
]
mse = np.array(mse)
# mse = mse - best_case
mse = -mse + baseline
mse = mse.T
plt.figure(1, figsize=(8, 7))
l0, = plt.plot(mse[0], 'h-', color='red')
l1, = plt.plot(mse[1], 'o-', color='orange')
l2, = plt.plot(mse[2], '^-', color='green')
l3, = plt.plot(mse[3], 'v-', color='purple')
l4, = plt.plot(mse[4], 'D-', color='blue')
plt.xlabel('Fourier divider', fontsize=font_size, family=font_type)
plt.ylabel('MSE score improvement', fontsize=font_size, family=font_type)
plt.ylim(-0.02, 0.23)
plt.legend(handles=[l0, l1, l2, l3, l4],
           labels=['pred_len=24', 'pred_len=48', 'pred_len=168', 'pred_len=336', 'pred_len=720'],
           loc='lower left', prop=font2)
plt.xticks([0, 1, 2, 3, 4], ['$2^0$', '$2^1$', '$2^2$', '$2^3$', '$2^4$'], fontsize=font_size-5, rotation=1)
plt.yticks([0, 0.05, 0.10, 0.15, 0.20], fontsize=font_size-5)

plt.savefig("/Users/marc-antoine/Documents/icml/fourier_divider.pdf", dpi=300, bbox_inches = 'tight')
plt.show()

# Sigmoid Temperature
mse = [
    [0.40, 0.53, 0.78, 1.11, 1.17],
    [0.42, 0.478, 0.79, 1.09, 1.12],
    [0.41, 0.48, 0.77, 1.04, 1.13],
    [0.41, 0.50, 0.80, 1.01, 1.10],
    [0.44, 0.51, 0.82, 1.02, 1.073],
]
mse = np.array(mse)
# mse = mse - best_case
mse = -mse + baseline
mse = mse.T
plt.figure(1, figsize=(8, 7))
l0, = plt.plot(mse[0], 'h-', color='red')
l1, = plt.plot(mse[1], 'o-', color='orange')
l2, = plt.plot(mse[2], '^-', color='green')
l3, = plt.plot(mse[3], 'v-', color='purple')
l4, = plt.plot(mse[4], 'D-', color='blue')
plt.xlabel('Sigmoid Temperature', fontsize=font_size, family=font_type)
plt.ylabel('MSE score improvement', fontsize=font_size, family=font_type)
plt.ylim(-0.02, 0.23)
plt.legend(handles=[l0, l1, l2, l3, l4],
           labels=['pred_len=24', 'pred_len=48', 'pred_len=168', 'pred_len=336', 'pred_len=720'],
           loc=0, prop=font2)
plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], fontsize=font_size-5, rotation=1)
plt.yticks([0, 0.05, 0.10, 0.15, 0.20], fontsize=font_size-5)

plt.savefig("/Users/marc-antoine/Documents/icml/temperature.pdf", dpi=300, bbox_inches = 'tight')
plt.show()

