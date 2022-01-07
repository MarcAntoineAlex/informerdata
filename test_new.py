from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
# from numpy import linspace
#
from data.data_loader import Dataset_ETT_hour
import time
import pandas as pd
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/Users/marc-antoine/Documents/S7/物理实验/UV3600/1.txt', sep=',')
print(data.columns)

font2 = {'weight': 'normal',
         'size': 14,
         }

x = data['NM'][:5000].to_numpy()
y = data['INTEN'][:5000].to_numpy()
x = x.astype(np.float)
y = y.astype(np.float)

plt.plot(x[:], y[:])
plt.xlabel('{}/nm'.format(chr(955)), font2)
plt.ylabel('U/V', font2)
plt.grid()
plt.xlim(500, 650)
plt.ylim(0)

plt.savefig('/Users/marc-antoine/Documents/temp/shangyong.jpg')
plt.show()
