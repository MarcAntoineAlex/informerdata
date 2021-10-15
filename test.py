from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from numpy import linspace

from data.data_loader import Dataset_ETT_hour
import time
import pandas as pd
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.L = nn.Linear(2, 4)
    def forward(self, x):
        return self.L(x)


def show(ori_func, ft, sampling_period=1):
    n = len(ori_func)
    interval = sampling_period / n
    # 绘制原始函数
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, sampling_period, interval), ori_func, 'black')
    plt.xlabel('Time'), plt.ylabel('Amplitude')
    # 绘制变换后的函数
    plt.subplot(2,1,2)
    frequency = np.arange(n / 200) / (n * interval)
    nfft = abs(ft[range(int(n / 200))] / n )
    plt.plot(frequency, nfft, 'red')
    plt.xlabel('Freq (Hz)'), plt.ylabel('Amp. Spectrum')
    plt.show()

data = pd.read_csv('/Users/marc-antoine/Downloads/travail/2_1.csv', sep=';')
temps = data['Temps'].to_numpy()
volt = data['EA1'].to_numpy()



from scipy.interpolate import make_interp_spline
xnew = np.linspace(temps.min(), temps.max(), 80000) #300 represents number of points to make between T.min and T.max
power_smooth = make_interp_spline(temps, volt)(xnew)
xnew = xnew*0.557

# plt.plot(xnew, power_smooth, linewidth=0.5)
# plt.grid()
# plt.xlim(0, 80)
# plt.xlabel('T/s')
# plt.ylabel('U/V')
# plt.savefig('/Users/marc-antoine/Documents/delta.jpg')
# plt.show()

after_idct = idct(power_smooth)
# plt.plot(np.reciprocal(x[200:500])*200000, y[200:500]/1000)
x = np.reciprocal(xnew[200:590])*100
y = after_idct[200:590]/3000
x_final, y_final = [], []
for i in range(len(x)-2):
    if y[i] <= y[i+1] and y[i+1] >= y[i+2] and y[i+1] > 0:
        x_final.append(x[i+1])
        y_final.append(y[i+1])
plt.plot(x_final, y_final)
plt.xlabel('{}/nm'.format(chr(955)))
plt.ylabel('U/V')
plt.grid()
plt.xlim(300, 900)

print(np.reciprocal(xnew)[np.argmax(y[200:500])+200]*100)
plt.savefig('/Users/marc-antoine/Documents/bao.jpg')
plt.show()

