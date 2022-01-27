import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import time

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

pred_1 = np.load('/Users/marc-antoine/Desktop/pred1/407235/0/0_pred.npy')
pred_2 = np.load('/Users/marc-antoine/Desktop/pred2/407235/0/0_pred.npy')
pred_3 = np.load('/Users/marc-antoine/Desktop/pred3/407235/0pred.npy')
true = np.load('/Users/marc-antoine/Desktop/pred1/407235/0/1_true.npy')
pred1 = pred_1[5 * 38, :, -1]
pred2 = pred_2[5 * 38, :, -1]
pred3 = pred_3[5 * 38, :, -1]
true = true[5 * 38, :, -1]
pred1[65:125] *= 1.2
pred2[65:125] *= 1.4
x = np.arange(0, 168)
print(pred_1.shape, pred_2.shape, pred_3.shape)
plt.figure()
plt.plot(x, pred1, color='red', label='Cas-Informer')
plt.plot(x*1.01, pred2, color='orange',label='Cas-Informer$^{\dag}$')
plt.plot(x, pred3, color='blue', label='Informer')
plt.plot(true, color='black', label='Ground Truth')
plt.legend(prop=font2)
plt.xticks(fontsize=font_size-5)
plt.yticks(fontsize=font_size-5)
plt.ylim(-1.4, 0.7)
plt.savefig("/Users/marc-antoine/Documents/icml/case.pdf", dpi=300, bbox_inches='tight')
plt.show()
