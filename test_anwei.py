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

#### FOURRIER
# # preprocessing
data = pd.read_csv('/Users/marc-antoine/Documents/S7/物理实验/gch/erhao.csv', sep=';')
temps = data['Temps'].to_numpy()[20:]
volt = data['EA1'].rolling(10).mean().to_numpy()[20:]
# volt *= 1.0293
volt -= volt.mean()
# temps = np.concatenate((temps, np.linspace(80, 90, 10000)))
# volt = np.concatenate((volt, volt[74980:79980]/2))
# volt = np.concatenate((volt, volt[74980:79980]/2))
font2 = {'weight': 'normal',
         'size': 14,
         }

# I(t)
plt.plot(temps, (volt+1.356)*1.0293-0.5508, linewidth=0.5, color='black')
plt.xlabel('T/s', font2)
plt.ylabel('I/cd', font2)
plt.grid()
plt.xlim(43, 80)
plt.ylim(0.5, 2)
plt.savefig('/Users/marc-antoine/Documents/S7/物理实验/gch/I(t).jpg')
plt.show()